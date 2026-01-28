import numpy as np
import torch


"""
RolloutBuffer for PPO (on-policy).

Purpose:
- Store a fixed number of transitions collected with the OLD policy
- Hold data needed for PPO updates
- No gradients, no learning logic

This buffer is cleared and recreated every PPO iteration.
"""


class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        """
        Parameters
        ----------
        size : int
            Number of environment steps to store (steps_per_iter)
        obs_dim : int
            Dimension of the observation/state space
        act_dim : int
            Dimension of the action space
        """

        # ----------------------------------------------------
        # Preallocate memory for rollout data
        # ----------------------------------------------------
        # Using NumPy arrays is efficient and simple.
        # Conversion to torch tensors happens only once
        # when training begins.
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)

        # Scalars per timestep
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)

        # Pointer to the current index in the buffer
        self.ptr = 0

    def store(self, obs, act, rew, done, logp, val):
        """
        Store one transition from the environment.

        This is called at every environment step during rollout.

        Parameters
        ----------
        obs : np.ndarray
            Observation (state) at time t
        act : np.ndarray
            Action sampled from the policy at time t
        rew : float
            Reward received after taking the action
        done : bool
            Whether the episode terminated at this step
        logp : float
            Log-probability of the action under the OLD policy
        val : float
            Value estimate V(s_t) under the OLD policy
        """

        # Store transition data at the current pointer location
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.logp[self.ptr] = logp
        self.val[self.ptr] = val

        # Advance pointer
        self.ptr += 1

    def get(self):
        """
        Return all stored rollout data as torch tensors.

        This is called ONCE per PPO iteration,
        after the rollout is complete.

        No shuffling happens here.
        PPO update logic handles batching and shuffling.
        """

        return dict(
            obs=torch.as_tensor(self.obs),
            act=torch.as_tensor(self.act),
            rew=torch.as_tensor(self.rew),
            done=torch.as_tensor(self.done),
            logp=torch.as_tensor(self.logp),
            val=torch.as_tensor(self.val),
        )



## this code is refined or written by chatgpt