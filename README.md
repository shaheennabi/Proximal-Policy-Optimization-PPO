# Proximal Policy Optimization (PPO)

A PyTorch implementation of the Proximal Policy Optimization algorithm, a state-of-the-art reinforcement learning method that balances performance and stability.

## Table of Contents

- [What is PPO?](#what-is-ppo)
- [How Does PPO Work?](#how-does-ppo-work)
- [Algorithm Overview](#algorithm-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Key Features](#key-features)

## What is PPO?

**Proximal Policy Optimization (PPO)** is a policy gradient reinforcement learning algorithm introduced by OpenAI in 2017. It's designed to be a simpler, more stable, and more sample-efficient alternative to previous policy gradient methods like A3C and TRPO (Trust Region Policy Optimization).

PPO has become one of the most popular algorithms in reinforcement learning due to its:
- **Stability**: The clipping mechanism prevents large policy updates
- **Sample Efficiency**: Reuses data effectively with multiple epochs of training
- **Ease of Implementation**: Simpler to implement and tune compared to TRPO
- **Performance**: Achieves state-of-the-art results on a wide range of tasks

## How Does PPO Work?

### Core Concept

PPO uses an **actor-critic** architecture where:
- **Actor**: A neural network that learns the policy (π) to decide what actions to take
- **Critic**: A neural network that estimates the value function (V) to evaluate how good a state is

### Training Process

1. **Rollout Phase**: The agent interacts with the environment, collecting trajectories (sequences of states, actions, and rewards)

2. **Advantage Estimation**: Using Generalized Advantage Estimation (GAE), compute how much better an action is compared to the baseline:
   $$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
   where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD-error

3. **Policy Update**: Update the policy using the PPO objective with clipping:
   $$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$
   
   where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio

4. **Value Update**: Update the value function to better predict returns:
   $$L^{VF}(\theta) = (V_\theta(s_t) - R_t)^2$$

5. **Entropy Bonus**: Add entropy regularization to encourage exploration:
   $$L^{ENT}(\theta) = -\beta \mathbb{E}[H(\pi_\theta)]$$

The clipping mechanism is the key innovation—it prevents the policy from changing too drastically in a single update, maintaining training stability.

## Algorithm Overview

```
for episode in range(num_episodes):
    # Collect experience
    trajectories = collect_rollout(env, policy, horizon)
    
    # Compute advantages using GAE
    advantages = compute_advantages(trajectories, value_function)
    
    # Multiple epochs of SGD on collected data
    for epoch in range(num_epochs):
        for minibatch in create_minibatches(trajectories):
            # Compute PPO loss
            loss = compute_ppo_loss(policy, minibatch, advantages)
            
            # Update policy and value function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate and log performance
    log_metrics()
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download the repository**:
   ```bash
   cd Proximal-Policy-Optimization-PPO-
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

- **torch**: Deep learning framework
- **gymnasium**: Environment toolkit (successor to gym)
- **numpy**: Numerical computing

## Project Structure

```
.
├── agent.py           # Actor-Critic network architecture
├── buffer.py          # Experience replay buffer for collecting trajectories
├── config.py          # Hyperparameter configuration
├── gae.py             # Generalized Advantage Estimation implementation
├── train.py           # Training loop and main training logic
├── update.py          # PPO update step and loss computation
├── main.py            # Entry point for running training
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### Key Files Explained

- **agent.py**: Defines the neural network architecture for the policy (actor) and value function (critic)
- **buffer.py**: Stores and manages collected experience trajectories during training
- **gae.py**: Implements Generalized Advantage Estimation for better advantage estimates
- **config.py**: Contains all hyperparameters (learning rate, discount factor, etc.)
- **train.py**: Implements the main training loop that orchestrates the PPO algorithm
- **update.py**: Contains the PPO loss function and optimization step

## Usage

### Basic Training

Run the training script:

```bash
python main.py
```

This will start training with the default configuration for 50 episodes.

### Custom Training

Modify the `main.py` file to customize training:

```python
from train import training_loop

training_loop(num_episodes=100)  # Train for 100 episodes
```

## Configuration

Hyperparameters can be adjusted in `config.py`:

```python
gamma = 0.99              # Discount factor
lam = 0.5                 # GAE lambda parameter
clip_eps = 0.2            # PPO clipping parameter
vf_coef = 0.5             # Value function loss coefficient
ent_coef = 0.0            # Entropy regularization coefficient
batch_size = 64           # Batch size for updates
```

### Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor (how much to value future rewards) |
| `lam` | 0.5 | GAE lambda (balance between bias and variance) |
| `clip_eps` | 0.2 | Clipping range for policy updates |
| `vf_coef` | 0.5 | Weight of value function loss |
| `ent_coef` | 0.0 | Weight for entropy bonus (exploration) |
| `batch_size` | 64 | Number of samples per gradient update |

## Key Features

- **PPO Clipping**: Prevents overly large policy updates
- **Generalized Advantage Estimation (GAE)**: Provides low-variance advantage estimates
- **Actor-Critic Architecture**: Combines policy and value function learning
- **Multi-epoch Training**: Reuses collected data efficiently
- **Entropy Regularization**: Encourages exploration during training

## References

- Original PPO Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- OpenAI Spinning Up: https://spinningup.openai.com/
- Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
