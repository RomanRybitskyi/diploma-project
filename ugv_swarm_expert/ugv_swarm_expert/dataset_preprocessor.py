from __future__ import annotations

import argparse
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TIME_COLUMN_CANDIDATES = ("timestamp", "time", "time_sec", "stamp", "t", "sec", "time_step")
COLUMN_ALIASES = {
    "x": ("x", "pos_x", "position_x", "pose_x"),
    "y": ("y", "pos_y", "position_y", "pose_y"),
    "theta": ("theta", "yaw", "heading"),
    "v": ("v", "linear_v", "linear_velocity", "linear_vel", "vel_x", "target_v"),
    "omega": ("omega", "w", "angular_w", "angular_velocity", "angular_vel", "target_w"),
}
KINEMATIC_COLUMNS = ("x", "y", "theta", "v", "omega")
LIDAR_SECTOR_COUNT = 36
LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 3.5
DEFAULT_GRID_FREQUENCY_HZ = 10.0
DEFAULT_ANOMALY_WINDOW_SEC = 0.5
DEFAULT_MAX_INTERPOLATION_GAP_SEC = 0.2


class DatasetPreprocessor:
    def __init__(
        self,
        output_frequency_hz: float = DEFAULT_GRID_FREQUENCY_HZ,
        anomaly_window_sec: float = DEFAULT_ANOMALY_WINDOW_SEC,
        max_abs_position_m: float = 1000.0,
        max_abs_linear_velocity_mps: float = 5.0,
        lidar_min_m: float = LIDAR_MIN_RANGE_M,
        lidar_max_m: float = LIDAR_MAX_RANGE_M,
        formation_time_column: str | None = None,
        max_interpolation_gap_sec: float = DEFAULT_MAX_INTERPOLATION_GAP_SEC,
        logger: logging.Logger | None = None,
    ):
        if output_frequency_hz <= 0.0:
            raise ValueError("output_frequency_hz must be positive.")
        if anomaly_window_sec < 0.0:
            raise ValueError("anomaly_window_sec must be non-negative.")
        if max_interpolation_gap_sec <= 0.0:
            raise ValueError("max_interpolation_gap_sec must be positive.")
        if lidar_min_m >= lidar_max_m:
            raise ValueError("lidar_min_m must be smaller than lidar_max_m.")

        self.output_frequency_hz = float(output_frequency_hz)
        self.grid_dt_sec = 1.0 / self.output_frequency_hz
        self.anomaly_window_sec = float(anomaly_window_sec)
        self.max_abs_position_m = float(max_abs_position_m)
        self.max_abs_linear_velocity_mps = float(max_abs_linear_velocity_mps)
        self.lidar_min_m = float(lidar_min_m)
        self.lidar_max_m = float(lidar_max_m)
        self.formation_time_column = formation_time_column
        self.max_interpolation_gap_sec = float(max_interpolation_gap_sec)
        self.logger = logger or LOGGER

    def preprocess(
        self,
        agent_csv_paths: Mapping[str, Path],
        output_csv_path: Path | None = None,
    ) -> pd.DataFrame:
        if not agent_csv_paths:
            raise ValueError("agent_csv_paths must contain at least one agent CSV.")

        cleaned_by_agent: dict[str, pd.DataFrame] = {}
        for agent_name, csv_path in agent_csv_paths.items():
            raw = pd.read_csv(csv_path)
            standardized = self.standardize_columns(raw)
            cleaned = self.clean_agent_frame(standardized, agent_name=agent_name)
            if cleaned.empty:
                raise ValueError(f"All rows were removed for agent '{agent_name}'.")
            cleaned_by_agent[agent_name] = cleaned

        synchronized = self.synchronize(cleaned_by_agent)
        if output_csv_path is not None:
            output_path = Path(output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            synchronized.to_csv(output_path, index=False)
            self.logger.info(
                "Saved synchronized clean dataset to %s (%d rows).", output_path, len(synchronized)
            )
        return synchronized

    def standardize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        time_column = self._resolve_time_column(frame.columns)
        canonical = pd.DataFrame(index=frame.index)
        canonical["time"] = pd.to_numeric(frame[time_column], errors="coerce")
        if time_column == "time_step":
            canonical["time"] = canonical["time"] * self.grid_dt_sec

        for canonical_name, aliases in COLUMN_ALIASES.items():
            source_column = self._find_first_column(frame.columns, aliases)
            if source_column is None:
                raise ValueError(f"Missing required column for '{canonical_name}'. Tried aliases: {aliases}.")
            canonical[canonical_name] = pd.to_numeric(frame[source_column], errors="coerce")

        lidar_columns = self._resolve_lidar_columns(frame.columns)
        if len(lidar_columns) != LIDAR_SECTOR_COUNT:
            raise ValueError(
                f"Expected {LIDAR_SECTOR_COUNT} LiDAR sector columns, found {len(lidar_columns)}."
            )
        for index, source_column in enumerate(lidar_columns, start=1):
            canonical[f"lidar_s{index}"] = pd.to_numeric(frame[source_column], errors="coerce")

        canonical = canonical.replace([np.inf, -np.inf], np.nan)
        canonical = canonical.dropna(subset=["time", *KINEMATIC_COLUMNS])
        canonical = canonical.sort_values("time", kind="mergesort")
        canonical = self._deduplicate_timestamps(canonical)
        return canonical.reset_index(drop=True)

    def clean_agent_frame(self, frame: pd.DataFrame, agent_name: str = "agent") -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        cleaned = frame.copy()
        lidar_columns = self._lidar_columns(cleaned.columns)
        cleaned.loc[:, lidar_columns] = self.clean_lidar_values(cleaned[lidar_columns])

        anomaly_mask = (
            cleaned["x"].abs().gt(self.max_abs_position_m)
            | cleaned["y"].abs().gt(self.max_abs_position_m)
            | cleaned["v"].abs().gt(self.max_abs_linear_velocity_mps)
        ).to_numpy(dtype=bool)
        anomaly_count = int(anomaly_mask.sum())
        drop_mask = self._anomaly_window_mask(cleaned["time"].to_numpy(dtype=np.float64), anomaly_mask)
        dropped_count = int(drop_mask.sum())

        if dropped_count > 0:
            self.logger.info(
                "Agent '%s': detected %d anomalous frame(s), dropped %d frame(s) including %.3fs stabilization windows.",
                agent_name,
                anomaly_count,
                dropped_count,
                self.anomaly_window_sec,
            )
        else:
            self.logger.info("Agent '%s': no physics anomalies detected.", agent_name)

        return cleaned.loc[~drop_mask].reset_index(drop=True)

    def synchronize(self, cleaned_by_agent: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        if not cleaned_by_agent:
            raise ValueError("cleaned_by_agent must contain at least one agent.")

        common_start = max(float(frame["time"].iloc[0]) for frame in cleaned_by_agent.values())
        common_end = min(float(frame["time"].iloc[-1]) for frame in cleaned_by_agent.values())
        if common_end < common_start:
            raise ValueError("Agent trajectories do not overlap in time after cleaning.")

        time_grid = self._make_time_grid(common_start, common_end)
        if time_grid.size == 0:
            raise ValueError("No synchronized 10 Hz grid points fall within the valid time range.")

        output = pd.DataFrame({"time": time_grid})
        valid_all_agents = np.ones(time_grid.shape, dtype=bool)

        for agent_name, frame in cleaned_by_agent.items():
            interpolated, valid_mask = self.interpolate_agent(frame, time_grid, agent_name)
            valid_all_agents &= valid_mask
            output = pd.concat([output, interpolated], axis=1)

        synchronized = output.loc[valid_all_agents].reset_index(drop=True)
        synchronized["time"] = np.round(synchronized["time"].to_numpy(dtype=np.float64), 3)

        if synchronized.empty:
            raise ValueError(
                "No synchronized rows remain. Consider increasing max_interpolation_gap_sec "
                "or checking anomaly-filtered trajectory coverage."
            )
        self.logger.info(
            "Synchronized %d agent(s) to %d clean 10 Hz row(s) from %.3fs to %.3fs.",
            len(cleaned_by_agent),
            len(synchronized),
            float(synchronized["time"].iloc[0]),
            float(synchronized["time"].iloc[-1]),
        )
        return synchronized

    def interpolate_agent(
        self,
        frame: pd.DataFrame,
        time_grid: np.ndarray,
        agent_name: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        source_time = frame["time"].to_numpy(dtype=np.float64)
        if source_time.size < 2:
            raise ValueError(f"Agent '{agent_name}' needs at least two samples for interpolation.")

        valid_mask = self._grid_gap_validity(source_time, time_grid)
        result: dict[str, np.ndarray] = {}

        for column in ("x", "y", "v", "omega"):
            result[f"{agent_name}_{column}"] = np.interp(
                time_grid,
                source_time,
                frame[column].to_numpy(dtype=np.float64),
            )

        result[f"{agent_name}_theta"] = interpolate_yaw(
            source_time,
            frame["theta"].to_numpy(dtype=np.float64),
            time_grid,
        )

        for lidar_column in self._lidar_columns(frame.columns):
            result[f"{agent_name}_{lidar_column}"] = np.interp(
                time_grid,
                source_time,
                frame[lidar_column].to_numpy(dtype=np.float64),
            )

        interpolated = pd.DataFrame(result)
        return interpolated, valid_mask

    def clean_lidar_values(self, lidar_frame: pd.DataFrame) -> pd.DataFrame:
        numeric = lidar_frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.lidar_max_m)
        return numeric.clip(lower=self.lidar_min_m, upper=self.lidar_max_m)

    def _resolve_time_column(self, columns: Iterable[str]) -> str:
        if self.formation_time_column is not None:
            if self.formation_time_column not in columns:
                raise ValueError(f"Configured time column '{self.formation_time_column}' is missing.")
            return self.formation_time_column

        return self._find_first_column(columns, TIME_COLUMN_CANDIDATES) or self._raise_missing_time_column()

    @staticmethod
    def _find_first_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
        normalized = {column.lower(): column for column in columns}
        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]
        return None

    @staticmethod
    def _raise_missing_time_column() -> str:
        raise ValueError(f"Missing time column. Tried: {TIME_COLUMN_CANDIDATES}.")

    @staticmethod
    def _resolve_lidar_columns(columns: Iterable[str]) -> list[str]:
        indexed_columns: list[tuple[int, str]] = []
        pattern = re.compile(r"^(?:lidar[_-]?s?|scan[_-]?s?|range[_-]?s?)(\d+)$", re.IGNORECASE)
        for column in columns:
            match = pattern.match(column)
            if match:
                indexed_columns.append((int(match.group(1)), column))

        if indexed_columns:
            indexed_columns.sort(key=lambda item: item[0])
            return [column for _, column in indexed_columns[:LIDAR_SECTOR_COUNT]]

        lidar_like = [column for column in columns if column.lower().startswith("lidar")]
        return sorted(lidar_like)[:LIDAR_SECTOR_COUNT]

    @staticmethod
    def _lidar_columns(columns: Iterable[str]) -> list[str]:
        return [
            f"lidar_s{index}" for index in range(1, LIDAR_SECTOR_COUNT + 1) if f"lidar_s{index}" in columns
        ]

    @staticmethod
    def _deduplicate_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
        if frame["time"].is_unique:
            return frame

        non_theta_columns = [column for column in frame.columns if column != "theta"]
        averaged = frame[non_theta_columns].groupby("time", as_index=False).mean(numeric_only=True)
        theta_grouped = frame.groupby("time")["theta"].agg(
            lambda values: math.atan2(np.sin(values).mean(), np.cos(values).mean())
        )
        return averaged.merge(theta_grouped.rename("theta"), on="time", how="inner")

    def _anomaly_window_mask(self, times: np.ndarray, anomaly_mask: np.ndarray) -> np.ndarray:
        if times.size != anomaly_mask.size:
            raise ValueError("times and anomaly_mask must have the same length.")
        if not anomaly_mask.any():
            return np.zeros(times.shape, dtype=bool)

        anomaly_times = times[anomaly_mask]
        starts = np.searchsorted(times, anomaly_times, side="left")
        ends = np.searchsorted(times, anomaly_times + self.anomaly_window_sec, side="right")
        delta = np.zeros(times.size + 1, dtype=np.int32)
        np.add.at(delta, starts, 1)
        np.add.at(delta, ends, -1)
        return np.cumsum(delta[:-1]) > 0

    def _make_time_grid(self, start: float, end: float) -> np.ndarray:
        step_count = int(np.floor((end - start) / self.grid_dt_sec + 1e-9)) + 1
        grid = start + np.arange(step_count, dtype=np.float64) * self.grid_dt_sec
        return np.round(grid, 3)

    def _grid_gap_validity(self, source_time: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        right_indices = np.searchsorted(source_time, time_grid, side="left")
        exact_last = right_indices == source_time.size
        right_indices = np.clip(right_indices, 0, source_time.size - 1)
        left_indices = np.clip(right_indices - 1, 0, source_time.size - 1)

        exact_match = np.isclose(source_time[right_indices], time_grid, atol=1e-9)
        left_indices = np.where(exact_match, right_indices, left_indices)
        left_indices = np.where(exact_last, source_time.size - 1, left_indices)
        right_indices = np.where(exact_last, source_time.size - 1, right_indices)

        surrounding_gap = source_time[right_indices] - source_time[left_indices]
        in_range = (time_grid >= source_time[0]) & (time_grid <= source_time[-1])
        return in_range & (surrounding_gap <= self.max_interpolation_gap_sec + 1e-9)


def interpolate_yaw(source_time: np.ndarray, source_yaw: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if source_time.ndim != 1 or source_yaw.ndim != 1 or target_time.ndim != 1:
        raise ValueError("source_time, source_yaw, and target_time must be one-dimensional arrays.")
    if source_time.size != source_yaw.size:
        raise ValueError("source_time and source_yaw must have the same length.")
    if source_time.size == 0:
        raise ValueError("source_time/source_yaw cannot be empty.")

    unwrapped = np.unwrap(source_yaw.astype(np.float64))
    interpolated = np.interp(target_time.astype(np.float64), source_time.astype(np.float64), unwrapped)
    return wrap_angle(interpolated)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def parse_agent_csv_args(agent_specs: Sequence[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in agent_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid agent CSV spec '{spec}'. Expected 'agent=/path/file.csv'.")
        agent_name, csv_path = spec.split("=", 1)
        agent_name = agent_name.strip()
        if not agent_name:
            raise ValueError(f"Invalid agent CSV spec '{spec}': empty agent name.")
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CSV path for agent '{agent_name}' does not exist: {path}")
        mapping[agent_name] = path
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and synchronize UGV swarm trajectory CSV logs.")
    parser.add_argument(
        "--agent-csv",
        dest="agent_csvs",
        action="append",
        required=True,
        help="Agent CSV mapping as 'agent_name=/path/to/raw.csv'. Repeat once per agent.",
    )
    parser.add_argument("--output", required=True, help="Path to the synchronized clean CSV output.")
    parser.add_argument(
        "--frequency", type=float, default=DEFAULT_GRID_FREQUENCY_HZ, help="Output grid frequency in Hz."
    )
    parser.add_argument("--time-column", default=None, help="Explicit source time column name.")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_INTERPOLATION_GAP_SEC,
        help="Maximum interpolation gap in seconds.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    preprocessor = DatasetPreprocessor(
        output_frequency_hz=args.frequency,
        formation_time_column=args.time_column,
        max_interpolation_gap_sec=args.max_gap,
    )
    preprocessor.preprocess(parse_agent_csv_args(args.agent_csvs), Path(args.output).expanduser())


if __name__ == "__main__":
    main()
