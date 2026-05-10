from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ugv_swarm_expert.actor_network import ActorNetwork
from ugv_swarm_expert.discriminator_network import DiscriminatorNetwork
from ugv_swarm_expert.gail_reward import compute_gail_reward
from ugv_swarm_expert.rollout_buffer import PPORolloutBuffer
from ugv_swarm_expert.trainer import MAGAILTrainer
from ugv_swarm_expert.weight_initializer import orthogonal_init

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


LOGGER = logging.getLogger(__name__)
STATE_SHAPE = (4, 41)
ACTION_DIM = 2


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4 * 41, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        orthogonal_init(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or tuple(states.shape[1:]) != STATE_SHAPE:
            raise ValueError(f"Critic input must have shape (Batch, 4, 41); got {tuple(states.shape)}.")
        return self.value_net(states).squeeze(-1)


class ExpertTensorDataset(Dataset):
    def __init__(self, path, device="cpu"):
        payload = torch.load(Path(path).expanduser(), map_location=device)
        if "states" not in payload or "actions" not in payload:
            raise KeyError("Expert tensor file must contain 'states' and 'actions'.")
        self.states = torch.as_tensor(payload["states"], dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(payload["actions"], dtype=torch.float32, device=device)
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError("Expert states/actions must share the same batch dimension.")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"states": self.states[index], "actions": self.actions[index]}


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class ExpertBatchIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def next(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MA-GAIL for a ROS 2/Gazebo UGV swarm.")
    parser.add_argument(
        "--expert-data", required=True, help="Path to expert .pt tensors from feature_engineer."
    )
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expert-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/ma_gail")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--disc-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--disc-epochs", type=int, default=3)
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Using device: %s", device)

    from ugv_swarm_expert.UGVSwarmEnv import UGVSwarmEnv

    env = UGVSwarmEnv(num_agents=args.num_agents, device=device)
    writer = SummaryWriter(args.log_dir) if SummaryWriter is not None else NullSummaryWriter()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    actor = ActorNetwork().to(device)
    critic = CriticNetwork().to(device)
    discriminator = DiscriminatorNetwork().to(device)

    buffer = PPORolloutBuffer(
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        state_shape=STATE_SHAPE,
        action_dim=ACTION_DIM,
        device=device,
    )
    expert_dataset = ExpertTensorDataset(args.expert_data, device="cpu")
    expert_loader = DataLoader(expert_dataset, batch_size=args.expert_batch_size, shuffle=True)
    expert_batches = ExpertBatchIterator(expert_loader)

    trainer = MAGAILTrainer(
        actor=actor,
        critic=critic,
        discriminator=discriminator,
        rollout_buffer=buffer,
        expert_dataloader=expert_loader,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        disc_lr=args.disc_lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        disc_epochs=args.disc_epochs,
        device=device,
    )

    try:
        for epoch in range(1, args.max_epochs + 1):
            states = env.reset().to(device)
            rollout_infos = []
            for _ in range(args.num_steps):
                with torch.no_grad():
                    flat_states = states.reshape(args.num_agents, *STATE_SHAPE)
                    action_dist = actor(flat_states)
                    actions = action_dist.sample().clamp(-1.0, 1.0)
                    log_probs = action_dist.log_prob(actions).sum(dim=-1)
                    values = critic(flat_states)

                next_states, dones, info = env.step(actions.detach().cpu().numpy())
                buffer.store(
                    state=flat_states,
                    action=actions,
                    log_prob=log_probs,
                    value=values,
                    reward=torch.zeros(args.num_agents, device=device),
                    done=torch.as_tensor(dones, dtype=torch.float32, device=device),
                )
                rollout_infos.append(info)
                states = next_states.to(device)

            final_state_frames = buffer.states[:, :, -1, :]
            gail_rewards = compute_gail_reward(
                discriminator, final_state_frames, buffer.actions, device=device
            )
            buffer.rewards.copy_(gail_rewards)
            with torch.no_grad():
                last_value = critic(states.reshape(args.num_agents, *STATE_SHAPE))
            buffer.compute_returns_and_advantages(last_value, gamma=args.gamma, gae_lambda=args.gae_lambda)

            expert_batch = move_batch_to_device(expert_batches.next(), device)
            generated_batch = {"states": buffer.states.detach(), "actions": buffer.actions.detach()}
            disc_metrics = trainer.update_discriminator(expert_batch, generated_batch)
            ppo_metrics = trainer.update_ppo(buffer, batch_size=args.batch_size)
            buffer.clear()

            mean_reward = float(gail_rewards.mean().detach().cpu())
            mean_collision = mean_info_value(rollout_infos, "collision")
            writer.add_scalar("reward/gail_mean", mean_reward, epoch)
            writer.add_scalar("safety/collision_rate", mean_collision, epoch)
            for name, value in {**disc_metrics, **ppo_metrics}.items():
                writer.add_scalar(f"loss/{name}", value, epoch)

            LOGGER.info(
                "Epoch %d/%d | reward=%.4f | actor=%.4f | critic=%.4f | disc=%.4f",
                epoch,
                args.max_epochs,
                mean_reward,
                ppo_metrics.get("actor_loss", 0.0),
                ppo_metrics.get("critic_loss", 0.0),
                disc_metrics.get("disc_loss", 0.0),
            )

            if epoch % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_dir, epoch, actor, critic, discriminator)
    finally:
        save_checkpoint(checkpoint_dir, "final", actor, critic, discriminator)
        writer.close()
        env.close()


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple | list):
        return type(batch)(move_batch_to_device(value, device) for value in batch)
    return batch


def mean_info_value(infos: list[dict[str, Any]], key: str) -> float:
    values = []
    for info in infos:
        if key in info:
            tensor = torch.as_tensor(info[key], dtype=torch.float32)
            values.append(float(tensor.mean()))
    return float(sum(values) / len(values)) if values else 0.0


def save_checkpoint(
    checkpoint_dir: Path, epoch, actor: nn.Module, critic: nn.Module, discriminator: nn.Module
) -> None:
    torch.save(actor.state_dict(), checkpoint_dir / f"actor_ep{epoch}.pth")
    torch.save(critic.state_dict(), checkpoint_dir / f"critic_ep{epoch}.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / f"discriminator_ep{epoch}.pth")


if __name__ == "__main__":
    main()
