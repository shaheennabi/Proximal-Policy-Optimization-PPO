from config import gamma, lam
import torch


"""
Generalized Advantage Estimation (GAE)

Purpose:
- Assign credit to actions across time
- Reduce variance compared to Monte Carlo returns
- Reduce bias compared to 1-step TD

GAE is a weighted sum of TD errors (delta),
controlled by lambda (λ).
"""


def compute_gae(rewards, values, dones, gamma=gamma, lam=lam):
    """
    Compute advantages and returns using GAE.

    Parameters
    ----------
    rewards : torch.Tensor, shape (T,)
        Rewards collected during the rollout
    values : torch.Tensor, shape (T,)
        Value estimates V(s_t) from the OLD policy
    dones : torch.Tensor, shape (T,)
        Episode termination flags (1 if done, 0 otherwise)
    gamma : float
        Discount factor (future reward importance)
    lam : float
        GAE lambda (bias-variance tradeoff)

    Returns
    -------
    advantages : torch.Tensor, shape (T,)
        Advantage estimates A_t
    returns : torch.Tensor, shape (T,)
        Target values for critic training (A_t + V(s_t))
    """

    # Number of timesteps in the rollout
    T = len(rewards)

    # Tensor to store computed advantages
    advantages = torch.zeros(T)

    # This variable holds A_{t+1} during backward recursion
    last_adv = 0.0

    # ----------------------------------------------------
    # Backward-time computation of GAE
    # ----------------------------------------------------
    # We iterate from the end of the trajectory to the beginning
    # because advantage at time t depends on advantage at t+1.
    for t in reversed(range(T)):

        # Bootstrap value from next state
        # If this is the last timestep, we use 0.0
        # (no future value available)
        next_value = values[t + 1] if t < T - 1 else 0.0

        # ------------------------------------------------
        # TD error (delta)
        # ------------------------------------------------
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        #
        # If episode terminated, we stop bootstrapping
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # ------------------------------------------------
        # GAE recursive formula
        # ------------------------------------------------
        # A_t = δ_t + γ * λ * A_{t+1}
        #
        # (1 - dones[t]) ensures we do not propagate advantage
        # across episode boundaries
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv

        # Store computed advantage
        advantages[t] = last_adv

    # ----------------------------------------------------
    # Compute returns (critic targets)
    # ----------------------------------------------------
    # R_t = A_t + V(s_t)
    #
    # This is what the value function is trained to predict
    returns = advantages + values

    # ----------------------------------------------------
    # Advantage normalization
    # ----------------------------------------------------
    # This is critical for PPO stability.
    #
    # Normalizing advantages:
    # - improves optimization conditioning
    # - prevents large policy updates
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns



## this code is refined or written by chatgpt