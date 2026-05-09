from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

STATE_WINDOW_SIZE = 4
STATE_FEATURE_COUNT = 41
ACTION_FEATURE_COUNT = 2
LIDAR_SECTOR_COUNT = 36

TB3_MIN_LINEAR_MPS = 0.0
TB3_MAX_LINEAR_MPS = 0.22
TB3_MIN_ANGULAR_RADPS = -2.84
TB3_MAX_ANGULAR_RADPS = 2.84
MIN_HEADING_RAD = -math.pi
MAX_HEADING_RAD = math.pi
FORMATION_ERROR_MIN_M = -2.0
FORMATION_ERROR_MAX_M = 2.0
LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 3.5

EPISODE_COLUMN_CANDIDATES = ("episode_id", "episode", "trajectory_id", "traj_id", "run_id")
TIME_COLUMN_CANDIDATES = ("time", "timestamp", "time_sec", "stamp", "t")


@dataclass(frozen=True)
class EngineeredDatasetArrays:
    states: np.ndarray
    actions: np.ndarray
    agent_names: np.ndarray
    episode_ids: np.ndarray
    augmented_flags: np.ndarray


class FeatureEngineer:
    def __init__(
        self,
        leader_name: str = "leader",
        follower_names=None,
        target_offsets=None,
        sequence_length: int = STATE_WINDOW_SIZE,
        augment_mirror: bool = True,
        logger: logging.Logger | None = None,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        self.leader_name = leader_name
        self.follower_names = list(follower_names) if follower_names is not None else None
        self.target_offsets = dict(target_offsets or {})
        self.sequence_length = int(sequence_length)
        self.augment_mirror = bool(augment_mirror)
        self.logger = logger or LOGGER

    def transform(self, frame: pd.DataFrame) -> EngineeredDatasetArrays:
        if frame.empty:
            raise ValueError("Input DataFrame is empty.")

        working = frame.copy()
        follower_names = self.follower_names or self.infer_follower_names(working, self.leader_name)
        if not follower_names:
            raise ValueError("No follower names were provided or inferred.")

        episode_column = self._resolve_episode_column(working.columns)
        if episode_column is None:
            working = working.assign(__episode_id__=0)
            episode_column = "__episode_id__"

        time_column = self._resolve_time_column(working.columns)
        sort_columns = [episode_column] + ([time_column] if time_column is not None else [])
        working = working.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_agent_names: list[np.ndarray] = []
        all_episode_ids: list[np.ndarray] = []
        all_augmented_flags: list[np.ndarray] = []

        for follower_name in follower_names:
            target_offset = np.asarray(self.target_offsets.get(follower_name, (-0.7, 0.0)), dtype=np.float32)
            for episode_id, episode_frame in working.groupby(episode_column, sort=False):
                if episode_frame.empty:
                    continue
                raw_states, raw_actions = self._extract_episode_features(
                    episode_frame, follower_name, target_offset
                )
                windows, actions = self._normalize_and_stack(raw_states, raw_actions)
                all_states.append(windows)
                all_actions.append(actions)
                all_agent_names.append(np.full(actions.shape[0], follower_name, dtype=object))
                all_episode_ids.append(np.full(actions.shape[0], episode_id, dtype=object))
                all_augmented_flags.append(np.zeros(actions.shape[0], dtype=bool))

                if self.augment_mirror:
                    mirrored_states, mirrored_actions = self.mirror_raw_features(raw_states, raw_actions)
                    mirror_windows, mirror_actions = self._normalize_and_stack(
                        mirrored_states, mirrored_actions
                    )
                    all_states.append(mirror_windows)
                    all_actions.append(mirror_actions)
                    all_agent_names.append(np.full(mirror_actions.shape[0], follower_name, dtype=object))
                    all_episode_ids.append(np.full(mirror_actions.shape[0], episode_id, dtype=object))
                    all_augmented_flags.append(np.ones(mirror_actions.shape[0], dtype=bool))

        if not all_states:
            raise ValueError("No samples were generated from the provided DataFrame.")

        states = np.concatenate(all_states, axis=0).astype(np.float32, copy=False)
        actions = np.concatenate(all_actions, axis=0).astype(np.float32, copy=False)
        result = EngineeredDatasetArrays(
            states=states,
            actions=actions,
            agent_names=np.concatenate(all_agent_names, axis=0),
            episode_ids=np.concatenate(all_episode_ids, axis=0),
            augmented_flags=np.concatenate(all_augmented_flags, axis=0),
        )
        self.logger.info(
            "Generated %d samples for %d follower(s); augmentation=%s.",
            result.states.shape[0],
            len(follower_names),
            self.augment_mirror,
        )
        return result

    def _extract_episode_features(
        self,
        episode_frame: pd.DataFrame,
        follower_name: str,
        target_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        leader_x = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, self.leader_name, "x")
        )
        leader_y = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, self.leader_name, "y")
        )
        leader_theta = wrap_angle(
            self._numeric_column(
                episode_frame, self._resolve_column(episode_frame, self.leader_name, "theta")
            )
        )

        follower_x = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, follower_name, "x")
        )
        follower_y = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, follower_name, "y")
        )
        follower_theta = wrap_angle(
            self._numeric_column(episode_frame, self._resolve_column(episode_frame, follower_name, "theta"))
        )
        follower_v = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, follower_name, "v")
        )
        follower_omega = self._numeric_column(
            episode_frame, self._resolve_column(episode_frame, follower_name, "omega")
        )
        lidar = self._extract_lidar_matrix(episode_frame, follower_name)

        dx_global = follower_x - leader_x
        dy_global = follower_y - leader_y
        cos_theta = np.cos(leader_theta)
        sin_theta = np.sin(leader_theta)
        local_x = cos_theta * dx_global + sin_theta * dy_global
        local_y = -sin_theta * dx_global + cos_theta * dy_global
        formation_error = np.column_stack((local_x - target_offset[0], local_y - target_offset[1]))

        raw_states = np.column_stack(
            (follower_v, follower_omega, follower_theta, formation_error, lidar)
        ).astype(np.float32)
        action_v, action_omega = self._extract_action_columns(episode_frame, follower_name)
        raw_actions = np.column_stack((action_v, action_omega)).astype(np.float32)
        return raw_states, raw_actions

    def _normalize_and_stack(
        self, raw_states: np.ndarray, raw_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_states = self.normalize_states(raw_states)
        normalized_actions = self.normalize_actions(raw_actions)
        return self.create_sliding_windows(normalized_states, self.sequence_length), normalized_actions

    @staticmethod
    def normalize_states(raw_states: np.ndarray) -> np.ndarray:
        states = np.asarray(raw_states, dtype=np.float32).copy()
        if states.ndim != 2 or states.shape[1] != STATE_FEATURE_COUNT:
            raise ValueError(f"raw_states must have shape (N, {STATE_FEATURE_COUNT}).")

        states[:, 2] = wrap_angle(states[:, 2])
        fill_values = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, *([LIDAR_MAX_RANGE_M] * LIDAR_SECTOR_COUNT)],
            dtype=np.float32,
        )
        states = np.where(np.isnan(states), fill_values, states)
        lower = np.asarray(
            [
                TB3_MIN_LINEAR_MPS,
                TB3_MIN_ANGULAR_RADPS,
                MIN_HEADING_RAD,
                FORMATION_ERROR_MIN_M,
                FORMATION_ERROR_MIN_M,
                *([LIDAR_MIN_RANGE_M] * LIDAR_SECTOR_COUNT),
            ],
            dtype=np.float32,
        )
        upper = np.asarray(
            [
                TB3_MAX_LINEAR_MPS,
                TB3_MAX_ANGULAR_RADPS,
                MAX_HEADING_RAD,
                FORMATION_ERROR_MAX_M,
                FORMATION_ERROR_MAX_M,
                *([LIDAR_MAX_RANGE_M] * LIDAR_SECTOR_COUNT),
            ],
            dtype=np.float32,
        )
        clipped = np.clip(states, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def normalize_actions(raw_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_FEATURE_COUNT:
            raise ValueError(f"raw_actions must have shape (N, {ACTION_FEATURE_COUNT}).")

        actions = np.nan_to_num(actions, nan=0.0, posinf=TB3_MAX_ANGULAR_RADPS, neginf=TB3_MIN_ANGULAR_RADPS)
        linear = np.clip(actions[:, 0], TB3_MIN_LINEAR_MPS, TB3_MAX_LINEAR_MPS)
        angular = np.clip(actions[:, 1], TB3_MIN_ANGULAR_RADPS, TB3_MAX_ANGULAR_RADPS)
        linear_norm = 2.0 * (linear - TB3_MIN_LINEAR_MPS) / (TB3_MAX_LINEAR_MPS - TB3_MIN_LINEAR_MPS) - 1.0
        angular_norm = (
            2.0 * (angular - TB3_MIN_ANGULAR_RADPS) / (TB3_MAX_ANGULAR_RADPS - TB3_MIN_ANGULAR_RADPS) - 1.0
        )
        return np.column_stack((linear_norm, angular_norm)).astype(np.float32)

    @staticmethod
    def mirror_raw_features(raw_states: np.ndarray, raw_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(raw_states, dtype=np.float32).copy()
        actions = np.asarray(raw_actions, dtype=np.float32).copy()
        if states.ndim != 2 or states.shape[1] != STATE_FEATURE_COUNT:
            raise ValueError(f"raw_states must have shape (N, {STATE_FEATURE_COUNT}).")
        if actions.ndim != 2 or actions.shape[1] != ACTION_FEATURE_COUNT:
            raise ValueError(f"raw_actions must have shape (N, {ACTION_FEATURE_COUNT}).")

        states[:, 1] = -states[:, 1]
        states[:, 2] = wrap_angle(-states[:, 2])
        states[:, 4] = -states[:, 4]
        states[:, 5:] = states[:, 5:][:, ::-1].copy()
        actions[:, 1] = -actions[:, 1]
        return states, actions

    @staticmethod
    def create_sliding_windows(states: np.ndarray, sequence_length: int = STATE_WINDOW_SIZE) -> np.ndarray:
        normalized = np.asarray(states, dtype=np.float32)
        if normalized.ndim != 2 or normalized.shape[1] != STATE_FEATURE_COUNT:
            raise ValueError(f"states must have shape (N, {STATE_FEATURE_COUNT}).")
        if normalized.shape[0] == 0:
            raise ValueError("states must contain at least one row.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        prefix = np.repeat(normalized[:1], sequence_length - 1, axis=0)
        padded = np.concatenate((prefix, normalized), axis=0)
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=sequence_length, axis=0)
        return np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)

    @staticmethod
    def infer_follower_names(frame: pd.DataFrame, leader_name: str = "leader") -> list[str]:
        names = []
        for column in frame.columns:
            if column.endswith("_x"):
                candidate = column[:-2]
                if candidate != leader_name:
                    names.append(candidate)
        return sorted(set(names))

    @staticmethod
    def _resolve_episode_column(columns: Iterable[str]) -> str | None:
        lowered = {column.lower(): column for column in columns}
        for candidate in EPISODE_COLUMN_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        return None

    @staticmethod
    def _resolve_time_column(columns: Iterable[str]) -> str | None:
        lowered = {column.lower(): column for column in columns}
        for candidate in TIME_COLUMN_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        return None

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, agent_name: str, feature_name: str) -> str:
        candidates = [f"{agent_name}_{feature_name}"]
        if feature_name == "theta":
            candidates.extend([f"{agent_name}_yaw", f"{agent_name}_heading"])
        elif feature_name == "omega":
            candidates.extend([f"{agent_name}_w", f"{agent_name}_angular_velocity"])
        elif feature_name == "v":
            candidates.extend([f"{agent_name}_linear_velocity", f"{agent_name}_linear_v"])

        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        raise ValueError(f"Missing column for agent '{agent_name}' feature '{feature_name}'.")

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)

    @staticmethod
    def _extract_lidar_matrix(frame: pd.DataFrame, agent_name: str) -> np.ndarray:
        columns = [f"{agent_name}_lidar_s{index}" for index in range(1, LIDAR_SECTOR_COUNT + 1)]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing LiDAR columns for agent '{agent_name}': {missing[:3]}...")
        lidar = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        lidar = np.nan_to_num(
            lidar, nan=LIDAR_MAX_RANGE_M, posinf=LIDAR_MAX_RANGE_M, neginf=LIDAR_MIN_RANGE_M
        )
        return np.clip(lidar, LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M).astype(np.float32)

    @staticmethod
    def _extract_action_columns(frame: pd.DataFrame, follower_name: str) -> tuple[np.ndarray, np.ndarray]:
        linear_candidates = (
            f"{follower_name}_target_v",
            f"{follower_name}_cmd_v",
            f"{follower_name}_v_cmd",
            f"{follower_name}_v",
        )
        angular_candidates = (
            f"{follower_name}_target_w",
            f"{follower_name}_target_omega",
            f"{follower_name}_cmd_w",
            f"{follower_name}_cmd_omega",
            f"{follower_name}_w_cmd",
            f"{follower_name}_omega",
            f"{follower_name}_w",
        )
        linear_column = next((column for column in linear_candidates if column in frame.columns), None)
        angular_column = next((column for column in angular_candidates if column in frame.columns), None)
        if linear_column is None or angular_column is None:
            raise ValueError(f"Missing target action columns for follower '{follower_name}'.")
        return (
            pd.to_numeric(frame[linear_column], errors="coerce").to_numpy(dtype=np.float32),
            pd.to_numeric(frame[angular_column], errors="coerce").to_numpy(dtype=np.float32),
        )


class UGVSwarmDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame | str | Path | EngineeredDatasetArrays,
        leader_name: str = "leader",
        follower_names=None,
        target_offsets=None,
        sequence_length: int = STATE_WINDOW_SIZE,
        augment_mirror: bool = True,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        if isinstance(data, EngineeredDatasetArrays):
            arrays = data
        else:
            frame = pd.read_csv(data) if isinstance(data, str | Path) else data
            engineer = FeatureEngineer(
                leader_name=leader_name,
                follower_names=follower_names,
                target_offsets=target_offsets,
                sequence_length=sequence_length,
                augment_mirror=augment_mirror,
            )
            arrays = engineer.transform(frame)

        self.states = torch.as_tensor(arrays.states, dtype=self.dtype, device=self.device)
        self.actions = torch.as_tensor(arrays.actions, dtype=self.dtype, device=self.device)
        self.agent_names = arrays.agent_names
        self.episode_ids = arrays.episode_ids
        self.augmented_flags = arrays.augmented_flags

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((np.asarray(angle, dtype=np.float32) + np.pi) % (2.0 * np.pi)) - np.pi


def parse_target_offsets(specs=None) -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    for spec in specs or []:
        if "=" not in spec or "," not in spec:
            raise ValueError(f"Invalid target offset '{spec}'. Expected 'agent=dx,dy'.")
        agent, values = spec.split("=", 1)
        dx_text, dy_text = values.split(",", 1)
        offsets[agent.strip()] = (float(dx_text), float(dy_text))
    return offsets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert clean UGV swarm CSV data into PyTorch tensors.")
    parser.add_argument("--input", required=True, help="Clean synchronized CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file containing states/actions tensors.")
    parser.add_argument("--leader", default="leader", help="Leader column prefix.")
    parser.add_argument(
        "--followers", nargs="+", default=None, help="Follower column prefixes. Inferred if omitted."
    )
    parser.add_argument(
        "--target-offset",
        action="append",
        default=None,
        help="Target offset as 'agent=dx,dy'. Repeat per follower.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable symmetric mirroring augmentation.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    engineer = FeatureEngineer(
        leader_name=args.leader,
        follower_names=args.followers,
        target_offsets=parse_target_offsets(args.target_offset),
        augment_mirror=not args.no_augment,
    )
    arrays = engineer.transform(pd.read_csv(Path(args.input).expanduser()))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "states": torch.from_numpy(arrays.states),
            "actions": torch.from_numpy(arrays.actions),
            "agent_names": arrays.agent_names.tolist(),
            "episode_ids": arrays.episode_ids.tolist(),
            "augmented_flags": arrays.augmented_flags.tolist(),
        },
        output_path,
    )
    LOGGER.info("Saved engineered tensors to %s", output_path)


if __name__ == "__main__":
    main()
