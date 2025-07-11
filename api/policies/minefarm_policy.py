import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_logits = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

    def get_action(self, obs: np.ndarray, action: torch.Tensor = None):
        """
        Get an action from the policy, either by sampling or using a provided action.
        Handles both single observations and batches.
        """
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs
        
        # Add a batch dimension if it's a single observation
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        dist, value = self.forward(obs_t)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)

        # If it was a single observation, return single items
        if obs_t.shape[0] == 1:
            return action.item(), log_prob.item(), value.item()
        
        return action, log_prob, value 