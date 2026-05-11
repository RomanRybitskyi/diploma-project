# UGV Swarm Expert Data Collector

ROS 2 Python package containing `expert_data_collector`, a deterministic leader-follower algorithmic expert for collecting MA-GAIL demonstrations.

## Build

From the workspace root:

```bash
colcon build --packages-select ugv_swarm_expert
source install/setup.zsh
```

## Run

```bash
ros2 run ugv_swarm_expert expert_data_collector \
  --ros-args \
  -p leader_name:=leader \
  -p follower_names:="['tb3_1', 'tb3_2']" \
  -p formation_distance:=0.7 \
  -p output_dir:=~/ugv_swarm_expert_data
```

The node runs at 10 Hz, subscribes to `/{agent}/odom` and `/{agent}/scan`, publishes follower commands to `/{agent}/cmd_vel`, and writes one CSV file per follower.

## Useful Parameters

- `leader_name`: leader namespace/name. Default: `leader`.
- `follower_names`: follower namespace/name list. Default: `['tb3_1', 'tb3_2']`.
- `formation_distance`: default spacing for generated column offsets. Default: `0.7` m.
- `formation_offsets`: explicit offsets as `dx,dy;dx,dy`, one per follower. Example for a wedge: `-0.7,0.5;-0.7,-0.5`.
- `odom_topic_template`: default `/{agent}/odom`.
- `scan_topic_template`: default `/{agent}/scan`.
- `cmd_vel_topic_template`: default `/{agent}/cmd_vel`.
- `publish_commands`: publish expert `Twist` commands. Default: `true`.
- `output_dir`: CSV output directory. Default: `~/ugv_swarm_expert_data`.
- `max_data_age_sec`: freshness threshold for latest-message synchronization. Set `0.0` to disable. Default: `0.5`.

## CSV Format

Each follower CSV has exactly:

```text
time_step,pos_x,pos_y,yaw,rel_dist_lead,rel_ang_lead,lidar_s1,...,lidar_s36,target_v,target_w
```

## State Processor

`ugv_swarm_expert.state_processor.StateProcessor` converts follower odometry, leader odometry, and follower LiDAR into the PPO actor input tensor.

```python
from ugv_swarm_expert.state_processor import StateProcessor

processor = StateProcessor(target_offset=(-0.7, 0.0), device="cpu")
state = processor.process(follower_odom, leader_odom, follower_scan)
assert tuple(state.shape) == (4, 41)
```

The returned tensor is min-max normalized to `[0, 1]`, temporally padded on the first frame, and emitted on the configured CPU/CUDA device.

## Dataset Preprocessor

`ugv_swarm_expert.dataset_preprocessor.DatasetPreprocessor` cleans raw ROS/Gazebo CSV trajectories and synchronizes all agents to a strict 10 Hz grid.

```bash
ros2 run ugv_swarm_expert dataset_preprocessor \
  --agent-csv leader=/path/to/leader_raw.csv \
  --agent-csv tb3_1=/path/to/tb3_1_raw.csv \
  --agent-csv tb3_2=/path/to/tb3_2_raw.csv \
  --output /path/to/clean_swarm_dataset.csv
```

The preprocessor:

- drops Gazebo physics anomaly frames and the following 0.5 s stabilization window;
- clips all 36 LiDAR sectors to `[0.12, 3.5]` meters;
- interpolates `x`, `y`, `theta`, `v`, and `omega` onto a shared 10 Hz time grid;
- unwraps yaw before interpolation to avoid `-pi`/`pi` discontinuity artifacts;
- writes a wide CSV with one `time` column and agent-prefixed synchronized features.

## Feature Engineering

`ugv_swarm_expert.feature_engineer.UGVSwarmDataset` converts the clean synchronized CSV into PyTorch samples for the PPO actor.

```python
from ugv_swarm_expert.feature_engineer import UGVSwarmDataset

dataset = UGVSwarmDataset(
    "/path/to/clean_swarm_dataset.csv",
    leader_name="leader",
    follower_names=["tb3_1", "tb3_2"],
    target_offsets={"tb3_1": (-0.7, 0.0), "tb3_2": (-1.4, 0.0)},
)
state_sequence, action = dataset[0]
assert tuple(state_sequence.shape) == (4, 41)
assert tuple(action.shape) == (2,)
```

Offline tensor export is also available:

```bash
ros2 run ugv_swarm_expert feature_engineer \
  --input /path/to/clean_swarm_dataset.csv \
  --output /path/to/actor_training_tensors.pt \
  --leader leader \
  --followers tb3_1 tb3_2 \
  --target-offset tb3_1=-0.7,0.0 \
  --target-offset tb3_2=-1.4,0.0
```

The module uses hardware-based min-max normalization, symmetric mirroring augmentation, and 4-step padded sliding windows.

## Discriminator Network

`ugv_swarm_expert.discriminator_network.DiscriminatorNetwork` implements the centralized MA-GAIL discriminator used during CTDE training.

```python
import torch
from ugv_swarm_expert.discriminator_network import DiscriminatorNetwork

discriminator = DiscriminatorNetwork()
joint_state_action = torch.rand(32, 3, 43)  # Batch, agents, state+action
prob_expert = discriminator(joint_state_action)
assert tuple(prob_expert.shape) == (32, 1)
```

The network uses shared local agent encoding, batch-first multi-head self-attention, global max pooling over agents, and a sigmoid evaluator.

## ROS 2 / Gazebo Environment Wrapper

`ugv_swarm_expert.UGVSwarmEnv.UGVSwarmEnv` provides a Gym-style bridge between PPO policies and the ROS 2 TurtleBot swarm.

```python
import numpy as np
from ugv_swarm_expert.UGVSwarmEnv import UGVSwarmEnv

env = UGVSwarmEnv(num_agents=3)
states = env.reset()                       # shape: (3, 4, 41)
actions = np.zeros((3, 2), dtype=np.float32)  # normalized [-1, 1]
next_states, dones, info = env.step(actions)
env.close()
```

The wrapper publishes `/ugv_i/cmd_vel`, subscribes to `/ugv_i/odom` and `/ugv_i/scan`, un-normalizes actor outputs to TurtleBot limits, enforces a 10 Hz control period, and uses `/reset_simulation` for episode resets.

## PPO Rollout Buffer

`ugv_swarm_expert.rollout_buffer.PPORolloutBuffer` stores vectorized multi-agent PPO trajectories before an update.

```python
from ugv_swarm_expert.rollout_buffer import PPORolloutBuffer

buffer = PPORolloutBuffer(num_steps=128, num_agents=3, state_shape=(4, 41), action_dim=2, device="cpu")
# buffer.store(states, actions, log_probs, values, gail_rewards, dones)
# buffer.compute_returns_and_advantages(last_value)
for batch in buffer.get_batches(batch_size=64):
    states = batch["states"]
    actions = batch["actions"]
```

The buffer keeps rollout tensors as `(T, N, ...)`, computes GAE over time with vectorized agent operations, then flattens to `(T * N, ...)` mini-batches for PPO.

## GAIL Surrogate Reward

`ugv_swarm_expert.gail_reward.compute_gail_reward` converts centralized discriminator probabilities into cooperative per-agent rewards.

```python
from ugv_swarm_expert.gail_reward import compute_gail_reward

# states: (Batch, N, 41), actions: (Batch, N, 2)
rewards = compute_gail_reward(discriminator, states, actions, device="cuda")
assert tuple(rewards.shape) == (states.shape[0], states.shape[1])
```

The function concatenates state/action features to `(Batch, N, 43)`, runs the discriminator under `torch.no_grad()`, clamps probabilities before `-log(1 - D)`, and broadcasts the global swarm reward across all agents.

## MA-GAIL Trainer

`ugv_swarm_expert.trainer.MAGAILTrainer` coordinates the adversarial discriminator update and PPO actor/critic update.

```python
from ugv_swarm_expert.trainer import MAGAILTrainer

trainer = MAGAILTrainer(actor, critic, discriminator, rollout_buffer=buffer)
trainer.update_discriminator(expert_batch, generated_batch)
trainer.update_ppo(batch_size=128)
```

The discriminator update uses BCE targets (`expert=1`, `generated=0`). The PPO update normalizes rollout advantages globally, applies the clipped surrogate objective, adds an entropy bonus, updates the critic with MSE returns, and clips Actor/Critic gradients to `0.5`.

## Main MA-GAIL Training Loop

`ugv_swarm_expert.main` is the top-level orchestrator for ROS 2 rollout collection, GAIL reward computation, PPO updates, TensorBoard logging, and checkpoints.

```bash
ros2 run ugv_swarm_expert ma_gail_train \
  --expert-data /path/to/actor_training_tensors.pt \
  --num-agents 3 \
  --num-steps 2048 \
  --max-epochs 500 \
  --checkpoint-dir checkpoints \
  --log-dir runs/ma_gail
```

Each epoch performs rollout collection, discriminator-based surrogate reward computation, GAE, discriminator optimization, PPO Actor/Critic optimization, TensorBoard logging, and periodic checkpoint saving.
