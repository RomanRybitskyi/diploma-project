from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugv_swarm_expert.constants import LIDAR_SECTOR_COUNT, STATE_WINDOW_SIZE
from ugv_swarm_expert.data.feature_engineer import FeatureEngineer

LOGGER = logging.getLogger(__name__)

AGENT_ID_TO_NAME: dict[int, str] = {1: "leader", 2: "tb3_1", 3: "tb3_2"}

DEFAULT_TARGET_OFFSETS: dict[str, tuple[float, float]] = {
    "tb3_1": (-0.5, 0.5),
    "tb3_2": (-0.5, -0.5),
}

RAW_LIDAR_COLS: list[str] = [f"lidar_sec_{i}" for i in range(LIDAR_SECTOR_COUNT)]


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["timestamp", "agent_id"]).reset_index(drop=True)

    leader_df = df_sorted[df_sorted["agent_id"] == 1].reset_index(drop=True)
    follower1_df = df_sorted[df_sorted["agent_id"] == 2].reset_index(drop=True)
    follower2_df = df_sorted[df_sorted["agent_id"] == 3].reset_index(drop=True)

    n = len(leader_df)
    assert len(follower1_df) == n and len(follower2_df) == n, (
        f"Row count mismatch between agents: "
        f"leader={n}, tb3_1={len(follower1_df)}, tb3_2={len(follower2_df)}"
    )

    result = pd.DataFrame()
    result["time"] = leader_df["timestamp"].values
    result["episode_id"] = 0
    result["leader_x"] = leader_df["x"].values
    result["leader_y"] = leader_df["y"].values
    result["leader_theta"] = leader_df["theta"].values

    for follower_df, name in [(follower1_df, "tb3_1"), (follower2_df, "tb3_2")]:
        result[f"{name}_x"] = follower_df["x"].values
        result[f"{name}_y"] = follower_df["y"].values
        result[f"{name}_theta"] = follower_df["theta"].values

        v_prev = np.concatenate([[0.0], follower_df["v_cmd"].values[:-1]])
        omega_prev = np.concatenate([[0.0], follower_df["omega_cmd"].values[:-1]])
        result[f"{name}_v"] = v_prev
        result[f"{name}_omega"] = omega_prev

        result[f"{name}_target_v"] = follower_df["v_cmd"].values
        result[f"{name}_target_w"] = follower_df["omega_cmd"].values

        for i, raw_col in enumerate(RAW_LIDAR_COLS, start=1):
            result[f"{name}_lidar_s{i}"] = follower_df[raw_col].values

    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw dataset.csv to expert tensors (.pt) for offline MA-GAIL training."
    )
    parser.add_argument("--input", default="datasets/dataset.csv", help="Input long-format CSV path.")
    parser.add_argument(
        "--output",
        default="datasets/expert_tensors.pt",
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable symmetric mirroring augmentation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    input_path = Path(args.input).expanduser()
    LOGGER.info("Reading %s ...", input_path)
    df = pd.read_csv(input_path)
    LOGGER.info("Dataset shape: %s", df.shape)

    LOGGER.info("Pivoting long→wide format ...")
    wide_df = long_to_wide(df)
    LOGGER.info("Wide format: %d rows × %d columns", *wide_df.shape)

    engineer = FeatureEngineer(
        leader_name="leader",
        follower_names=["tb3_1", "tb3_2"],
        target_offsets=DEFAULT_TARGET_OFFSETS,
        sequence_length=STATE_WINDOW_SIZE,
        augment_mirror=not args.no_augment,
    )

    LOGGER.info("Running feature engineering (augment_mirror=%s) ...", not args.no_augment)
    arrays = engineer.transform(wide_df)

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
    LOGGER.info(
        "Saved %d expert samples  [states %s, actions %s]  →  %s",
        arrays.states.shape[0],
        tuple(arrays.states.shape),
        tuple(arrays.actions.shape),
        output_path,
    )


if __name__ == "__main__":
    main()
