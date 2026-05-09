Here is a comprehensive, highly detailed `README.md` for your project, written in English and structured to professional software engineering standards. It encapsulates the entire pipeline described in your diploma thesis, from data collection to inference.

---

# UGV Swarm Formation Control via MA-GAIL

## 📖 Project Overview

This repository contains the implementation of **Path Planning and Formation Control of an Unmanned Ground Vehicle (UGV) Swarm using Multi-Agent Generative Adversarial Imitation Learning (MA-GAIL)**.

The system leverages a decentralized neural network policy to enable a swarm of differential-drive UGVs (TurtleBot 3 Waffle Pi) to autonomously maintain geometric formations (e.g., "Column", "V-shape/Wedge") and navigate around dynamic/static obstacles. By utilizing an Imitation Learning approach, the system synthesizes complex control strategies from expert demonstrations without the need for manual, fragile reward function engineering.

---

## 🏗️ Architecture: Centralized Training, Decentralized Execution (CTDE)

The core architectural paradigm of this project is **CTDE**.

1. **Training Phase:** A centralized Discriminator acts as a global observer, possessing full access to the joint state-action space of the entire swarm. This allows it to evaluate swarm topology, formation consistency, and emergent coordination.


2. **Execution Phase (Inference):** The Discriminator is discarded. Each UGV runs its own Generator (Actor) autonomously, making decisions strictly based on local perceptual data (LiDAR, local odometry, relative target vector).



---

## 📊 1. Data Engineering & Collection

High-quality expert demonstrations are critical for Imitation Learning. Data is collected using an algorithmic expert based on a strict Leader-Follower deterministic model.

### State Space ($s_t$) & Action Space ($a_t$)

The system operates at 10 Hz ($\Delta t = 0.1s$). To capture system inertia and prevent control chatter, we utilize a **Sliding Window** of $k=4$ past frames (0.4 seconds).

* **Input Shape:** `(Batch, 4, 41)` per agent. The 41 features consist of:
* **Kinematics (3 features):** Linear velocity $v_i$, angular velocity $\omega_i$, and heading $\theta_i$.
* **Formation (2 features):** Relative offset $[\Delta x, \Delta y]$ to the target slot, calculated in the leader's local coordinate frame to guarantee translational and rotational invariance.
* **Perception (36 features):** 360 raw LiDAR rays are min-pooled into 36 sectors (10° each) to reduce dimensionality while preserving critical obstacle proximity.
* **Action Shape:** `(Batch, 2)`. Target linear velocity $v_{cmd} \in [0, 0.22]$ m/s and angular velocity $\omega_{cmd} \in [-2.84, 2.84]$ rad/s.



### Preprocessing Pipeline

* **Interpolation:** ROS 2 topics (`/odom`, `/scan`) are time-synchronized with $10^{-3}$s precision.
* **Normalization:** All features are scaled to $[0, 1]$ using Min-Max normalization to prevent gradient explosion.
* **Augmentation:** Trajectories are horizontally mirrored (inverting $\Delta y$, $\theta$, $\omega$, and reversing LiDAR sectors) to eliminate statistical bias and double the dataset size.

---

## 🧠 2. Model Architecture

### Generator / Actor ($\pi_\theta$)

Optimized via Proximal Policy Optimization (PPO), the Actor maps the continuous state space to actions using an Actor-Critic setup.

* **LiDAR Encoder:** 1D-CNN (16 filters, kernel 3) $\rightarrow$ ReLU $\rightarrow$ MaxPool(2) $\rightarrow$ Flatten $\rightarrow$ Dense(32).
* **Kinematic Branch:** Flatten $\rightarrow$ Dense(32).
* **Fusion MLP:** Concatenated 64-dim vector $\rightarrow$ Dense(256) $\rightarrow$ LayerNorm $\rightarrow$ ReLU $\rightarrow$ Dense(128) $\rightarrow$ LayerNorm $\rightarrow$ ReLU $\rightarrow$ Dense(64) $\rightarrow$ LayerNorm $\rightarrow$ ReLU.
* **Output:** Dense(2) with `Tanh` activation. Actions are sampled from $a_t \sim \mathcal{N}(\mu_\theta(s_t), \sigma^2)$ where $\sigma$ is a learned state-independent parameter.

### Discriminator ($D_\phi$)

The discriminator evaluates how well the generated trajectories match the expert data.

* **Input:** Joint state-action space `(Batch, N, 43)`.
* **Encoding:** Local MLP (`Dense(64, LeakyReLU)`) per agent.
* **Aggregation:** Multi-Head Attention followed by Global Max Pooling to extract N-invariant topology features.
* **Evaluator:** Dense(256) $\rightarrow$ LeakyReLU $\rightarrow$ Dense(128) $\rightarrow$ LeakyReLU $\rightarrow$ Dense(64) $\rightarrow$ LeakyReLU $\rightarrow$ Dense(1, Sigmoid).



---

## ⚙️ 3. Training Pipeline (PPO + MA-GAIL)

The model is trained using adversarial min-max optimization. The Generator does not receive environmental rewards; instead, it maximizes the surrogate reward provided by the Discriminator.

* **Reward Function:** $r(s_t, a_t) = -\log(1 - D_\phi(s_t, a_t))$. This provides an exponentially increasing penalty as the agent deviates from expert-like states.
* **Discriminator Loss:** Binary Cross-Entropy, separating expert data $P_E(s,a)$ ($D \approx 1$) from agent data $P_\pi(s,a)$ ($D \approx 0$).
* **Hyperparameters:**
* Optimizer: Adam
* Learning Rate: $3 \times 10^{-4}$
* Batch Size: 128
* Discount Factor ($\gamma$): 0.99
* Entropy Coeff ($\lambda$): 0.01 (Prevents mode collapse by encouraging exploration).





---

## 🚀 4. ROS 2 Integration & Control Flow

The software stack integrates PyTorch with ROS 2 and Gazebo physically realistic simulation.

* **Environment Wrapper:** Operates at 10 Hz. Subscribes to `/odom`, `/imu`, `/scan`. Buffers states into the $k=4$ sliding window.
* **Command Bridge:** Un-normalizes neural network outputs to hardware limits and publishes `geometry_msgs/Twist` to the `/cmd_vel` topics of the respective UGV namespaces.
* **Obstacle Avoidance Logic:** An explicit scalar feature $d_{min}$ (global LiDAR minimum) forces the network to learn a hierarchical priority: collision avoidance strictly supersedes formation accuracy.

---

## 📈 Performance & Metrics

Compared to classical Behaviour Cloning (BC), MA-GAIL demonstrates superior robustness:

* **Success Rate (SR):** 96.0% in dynamic obstacle scenarios (+24% over BC).
* **Mean Formation Error ($E_f$):** 0.039 meters (2.1x better than BC).
* **Smoothness Factor ($S_\omega$):** 0.38 (5.6x smoother control signals, preserving actuator lifespan).
* **Recovery Time ($T_{rec}$):** 1.5 seconds to restore formation after obstacle deformation.



---

## 🛠️ Setup & Installation

### Prerequisites

* Ubuntu 22.04 LTS or 24.04 LTS
* ROS 2 (Humble / Jazzy)
* NVIDIA GPU with CUDA 12.x support
* Docker & Docker Compose (Recommended)

*(Note: Add standard installation commands, `colcon build` instructions, and `docker-compose up` snippets here depending on your final repository structure).*
