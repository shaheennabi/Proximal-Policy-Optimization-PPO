from config import clip_eps, vf_coef, ent_coef, batch_size
import torch
import numpy as np


"""
PPO Update Logic

Purpose:
- Perform multiple epochs of PPO updates using on-policy data
- Optimize policy using clipped surrogate objective
- Train value function with MSE loss
- Optionally encourage exploration via entropy bonus

This file contains ALL gradient-based learning logic.
"""


def update(
    model,
    optimizer,
    data,
    advantages,
    returns,
    clip_eps=clip_eps,
    vf_coef=vf_coef,
    ent_coef=ent_coef,
    batch_size=batch_size,
    epochs=10,
):
    """
    Perform PPO updates.

    Parameters
    ----------
    model : ActorCritic
        Policy + value network
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    data : dict
        Rollout data (obs, act, logp, val)
    advantages : torch.Tensor, shape (T,)
        Advantage estimates from GAE
    returns : torch.Tensor, shape (T,)
        Target values for critic
    """

    obs = data["obs"]
    act = data["act"]
    logp_old = data["logp"]

    N = len(obs)

    # Diagnostics for monitoring PPO behavior
    kl_list = []
    clip_frac_list = []

    # ----------------------------------------------------
    # Multiple PPO epochs over the same rollout
    # ----------------------------------------------------
    # PPO reuses on-policy data for several gradient steps
    for _ in range(epochs):

        # Shuffle indices to create minibatches
        idx = torch.randperm(N)

        for start in range(0, N, batch_size):
            batch = idx[start : start + batch_size]

            # ------------------------------------------------
            # Policy evaluation (NEW policy)
            # ------------------------------------------------
            dist = model.policy(obs[batch])

            # Log-probabilities under current policy
            logp = dist.log_prob(act[batch]).sum(-1)

            # ------------------------------------------------
            # Importance sampling ratio
            # ------------------------------------------------
            # r_t = pi_new(a|s) / pi_old(a|s)
            ratio = torch.exp(logp - logp_old[batch])

            # ------------------------------------------------
            # PPO clipped surrogate objective
            # ------------------------------------------------
            surr1 = ratio * advantages[batch]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch]

            # Negative because we minimize loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # ------------------------------------------------
            # Value function loss (critic)
            # ------------------------------------------------
            value = model.value(obs[batch])
            value_loss = ((value - returns[batch]) ** 2).mean()

            # ------------------------------------------------
            # Entropy bonus (exploration)
            # ------------------------------------------------
            entropy = dist.entropy().sum(-1).mean()

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------
            loss = (
                policy_loss + vf_coef * value_loss - ent_coef * entropy)

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------------------------------------------------
            # Diagnostics (no gradients)
            # ------------------------------------------------
            approx_kl = (logp_old[batch] - logp).mean().item()
            clip_frac = (
                (torch.abs(ratio - 1.0) > clip_eps)
                .float()
                .mean()
                .item()
            )

            kl_list.append(approx_kl)
            clip_frac_list.append(clip_frac)

    # ----------------------------------------------------
    # Return metrics AFTER all updates
    # ----------------------------------------------------
    return {
        "kl": np.mean(kl_list),
        "clip_frac": np.mean(clip_frac_list),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }





## this code is mostly written or refined bychatgpt


