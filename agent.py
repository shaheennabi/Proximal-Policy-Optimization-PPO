import torch
import torch.nn as nn
from torch.distributions import Normal


"""
Actor-Critic network for PPO (continuous action spaces).

Key ideas:
- A shared backbone learns a representation of the state
- The actor outputs parameters of a probability distribution (Gaussian)
- The critic outputs a scalar value V(s)
- Policy and value share features but have separate heads
"""


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        # ----------------------------------------------------
        # Shared backbone
        # ----------------------------------------------------
        # This network maps observations (states) to a latent
        # feature representation.
        #
        # Both the policy (actor) and value function (critic)
        # use these shared features.
        #
        # This is standard in PPO and reduces sample complexity.
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # ----------------------------------------------------
        # Actor head (policy)
        # ----------------------------------------------------
        # This outputs the MEAN of a Gaussian distribution
        # for each action dimension.
        #
        # Shape:
        #   input : (batch_size, hidden_dim)
        #   output: (batch_size, act_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)

        # Log standard deviation (learnable parameter)
        #
        # - This is NOT conditioned on the state
        # - One std per action dimension
        # - We store log(std) for numerical stability
        #
        # Initial value = 0  -> std = exp(0) = 1
        # Meaning: start with high exploration
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # ----------------------------------------------------
        # Critic head (value function)
        # ----------------------------------------------------
        # Outputs a SINGLE scalar value V(s) for each state.
        #
        # Shape:
        #   input : (batch_size, hidden_dim)
        #   output: (batch_size, 1)
        #
        # We will later squeeze this to (batch_size,)
        self.v_head = nn.Linear(hidden_dim, 1)

    # --------------------------------------------------------
    # Policy function
    # --------------------------------------------------------
    def policy(self, obs):
        """
        Given observations, return a Gaussian policy distribution.

        This does NOT sample actions.
        It only constructs the distribution object.

        Steps:
        1. obs -> shared backbone -> features
        2. features -> mean (μ)
        3. log_std -> std (σ)
        4. return Normal(μ, σ)
        """
        features = self.shared(obs)

        # Mean of Gaussian policy
        mean = self.mean(features)

        # Standard deviation (ensure positivity via exp)
        std = torch.exp(self.log_std)

        # Create a Normal distribution object
        # This object knows how to:
        # - sample actions
        # - compute log-probabilities
        # - compute entropy
        return Normal(mean, std)

    # --------------------------------------------------------
    # Value function
    # --------------------------------------------------------
    def value(self, obs):
        """
        Given observations, return the scalar value V(s).

        Output shape:
        - Before squeeze: (batch_size, 1)
        - After squeeze : (batch_size,)

        PPO and GAE expect value tensors to be 1D.
        """
        features = self.shared(obs)

        # Remove the trailing singleton dimension
        return self.v_head(features).squeeze(-1)





### this code is refined or written by chatgpt...