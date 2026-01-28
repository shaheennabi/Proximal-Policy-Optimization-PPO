import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np

from agent import ActorCritic
from buffer import RolloutBuffer
from gae import compute_gae
from update import update


"""
Training loop for PPO.

This file is responsible ONLY for:
- Environment interaction
- Rollout collection
- Orchestrating GAE + PPO update

No learning math lives here.
"""


def training_loop(total_iterations, steps_per_iter=1200):
    """
    Parameters
    ----------
    total_iterations : int
        Number of PPO iterations (outer loop)
    steps_per_iter : int
        Number of environment steps collected per iteration
    """

    # ----------------------------------------------------
    # Environment setup
    # ----------------------------------------------------
    env = gym.make("Pendulum-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ----------------------------------------------------
    # Model and optimizer
    # ----------------------------------------------------
    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # ====================================================
    # Main PPO loop
    # ====================================================
    for itr in range(total_iterations):

        # ------------------------------------------------
        # Create a fresh rollout buffer
        # ------------------------------------------------
        # PPO is on-policy, so we discard old data
        buffer = RolloutBuffer(steps_per_iter, obs_dim, act_dim)

        # Reset environment at the start of each iteration
        obs, _ = env.reset()

        ep_return = 0.0
        ep_returns = []

        # =================================================
        # Rollout collection (OLD policy is frozen here)
        # =================================================
        for t in range(steps_per_iter):

            # Convert observation to tensor
            obs_t = torch.tensor(obs, dtype=torch.float32)

            # ------------------------------------------------
            # Policy inference (NO gradients)
            # ------------------------------------------------
            with torch.no_grad():
                dist = model.policy(obs_t)          # π_old(a|s)
                act = dist.sample()                  # sample action
                logp = dist.log_prob(act).sum(-1)   # log π_old(a|s)
                val = model.value(obs_t)            # V_old(s)

            # Step environment using sampled action
            next_state, reward, terminated, truncated, _ = env.step(act.numpy())
            done = terminated or truncated

            ep_return += reward

            # ------------------------------------------------
            # Store transition in rollout buffer
            # ------------------------------------------------
            buffer.store(
                obs=obs,
                act=act.numpy(),
                rew=reward,
                done=done,
                logp=logp.item(),
                val=val.item(),
            )

            # ------------------------------------------------
            # Move to next state
            # ------------------------------------------------
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_state

        # =================================================
        # PPO UPDATE PHASE (learning happens here)
        # =================================================

        # Get rollout data as torch tensors
        data = buffer.get()

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            data["rew"],
            data["val"],
            data["done"],
        )

        # Perform PPO update (multiple epochs, minibatches)
        metrics = update(
            model=model,
            optimizer=optimizer,
            data=data,
            advantages=advantages,
            returns=returns,
        )

        # ------------------------------------------------
        # Logging
        # ------------------------------------------------
        if itr % 10 == 0:
            print(
                f"Iter {itr:4d} | "
                f"Return {np.mean(ep_returns):8.1f} | "
                f"KL {metrics['kl']:.4f} | "
                f"Clip {metrics['clip_frac']:.2f} | "
                f"VLoss {metrics['value_loss']:.3f} | "
                f"Entropy {metrics['entropy']:.3f}"
            )


## this code is written or refined by chatgpt

## thanks for reading it