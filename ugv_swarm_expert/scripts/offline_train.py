from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugv_swarm_expert.constants import (
    STATE_FEATURE_COUNT,
    STATE_WINDOW_SIZE,
)
from ugv_swarm_expert.models.actor_network import ActorNetwork
from ugv_swarm_expert.models.critic_network import CriticNetwork
from ugv_swarm_expert.models.discriminator_network import DiscriminatorNetwork

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
except Exception:
    SummaryWriter = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger(__name__)
STATE_SHAPE = (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT)


class ExpertDataset(Dataset):
    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        payload = torch.load(Path(path).expanduser(), map_location=device, weights_only=True)
        if "states" not in payload or "actions" not in payload:
            raise KeyError("Expert tensor file must contain 'states' and 'actions'.")
        self.states = torch.as_tensor(payload["states"], dtype=torch.float32)
        self.actions = torch.as_tensor(payload["actions"], dtype=torch.float32)
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError("Expert states and actions must share the same batch dimension.")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


def _last_frame(states: torch.Tensor) -> torch.Tensor:
    return states[:, -1, :]


def _build_disc_input(state_frame: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    joint = torch.cat([state_frame, actions], dim=-1)
    return joint.unsqueeze(1)


def _save_checkpoint(
    checkpoint_dir: Path,
    tag: Any,
    actor: nn.Module,
    critic: nn.Module,
    discriminator: nn.Module,
) -> None:
    torch.save(actor.state_dict(), checkpoint_dir / f"actor_ep{tag}.pth")
    torch.save(critic.state_dict(), checkpoint_dir / f"critic_ep{tag}.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / f"discriminator_ep{tag}.pth")


class NullWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def train(args: argparse.Namespace) -> None:
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training device: %s", device)

    dataset = ExpertDataset(args.expert_data, device="cpu")
    LOGGER.info(
        "Expert dataset: %d samples (states %s, actions %s)",
        len(dataset),
        tuple(dataset.states.shape),
        tuple(dataset.actions.shape),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    actor = ActorNetwork().to(device)
    critic = CriticNetwork().to(device)
    discriminator = DiscriminatorNetwork().to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr, eps=1e-5)

    actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=args.epochs)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=args.epochs)
    disc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt, T_max=args.epochs)

    bce_loss = nn.BCELoss()

    log_dir = Path(args.log_dir)
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(log_dir))
        LOGGER.info("TensorBoard logs → %s", log_dir)
    else:
        writer = NullWriter()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        actor.train()
        critic.train()
        discriminator.train()

        total_bc = 0.0
        total_disc = 0.0
        total_critic = 0.0
        n_batches = 0

        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            B = states.size(0)

            actor_opt.zero_grad(set_to_none=True)
            dist = actor(states)
            bc_loss = -dist.log_prob(actions).sum(dim=-1).mean()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad(set_to_none=True)
            values = critic(states)
            target = torch.ones(B, device=device)
            critic_loss = F.mse_loss(values, target)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_opt.step()

            disc_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_actions = actor(states).mean.clamp(-1.0, 1.0)

            last_frame = _last_frame(states)
            expert_input = _build_disc_input(last_frame, actions)
            fake_input = _build_disc_input(last_frame, fake_actions)

            expert_preds = discriminator(expert_input)
            fake_preds = discriminator(fake_input)

            disc_loss = 0.5 * (
                bce_loss(expert_preds, torch.ones_like(expert_preds))
                + bce_loss(fake_preds, torch.zeros_like(fake_preds))
            )
            disc_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
            disc_opt.step()

            total_bc += bc_loss.item()
            total_disc += disc_loss.item()
            total_critic += critic_loss.item()
            n_batches += 1

        actor_sched.step()
        critic_sched.step()
        disc_sched.step()

        mean_bc = total_bc / max(n_batches, 1)
        mean_disc = total_disc / max(n_batches, 1)
        mean_critic = total_critic / max(n_batches, 1)

        LOGGER.info(
            "Epoch %4d/%d | BC=%.5f | critic=%.5f | disc=%.5f | lr=%.2e",
            epoch,
            args.epochs,
            mean_bc,
            mean_critic,
            mean_disc,
            actor_opt.param_groups[0]["lr"],
        )
        writer.add_scalar("loss/bc", mean_bc, epoch)
        writer.add_scalar("loss/disc", mean_disc, epoch)
        writer.add_scalar("loss/critic", mean_critic, epoch)

        if epoch % args.checkpoint_every == 0:
            _save_checkpoint(checkpoint_dir, epoch, actor, critic, discriminator)
            LOGGER.info("  → checkpoint saved (epoch %d)", epoch)

    _save_checkpoint(checkpoint_dir, "final", actor, critic, discriminator)
    LOGGER.info("Final models saved to %s", checkpoint_dir)
    writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline MA-GAIL training (Behavioral Cloning + Discriminator)."
    )
    parser.add_argument(
        "--expert-data",
        required=True,
        help="Path to expert .pt tensors (from prepare_expert_data.py).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--disc-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-dir", default="checkpoints/offline")
    parser.add_argument("--log-dir", default="runs/offline_magail")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    train(args)


if __name__ == "__main__":
    main()
