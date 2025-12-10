"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.

    3 things to try here:

    1. adding gaussian noise during training
    2. adding dropout
    3. adding full-token masking during training
    """
    def __init__(self, llm_dim: int, proprio_dim: int, noise_std=0, dropout_prob=0.3, mask_prob=0.2) -> None: # noise_std=0.05
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

        self.noise_std = noise_std

        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.mask_prob = mask_prob
        
        self.mask_token = nn.Parameter(torch.randn(1, llm_dim))


    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.dropout1(projected_features)
        projected_features = self.fc2(projected_features)
        projected_features = self.dropout2(projected_features)

        if self.training and torch.rand(1).item() < self.mask_prob:
            return self.mask_token.expand(proprio.size(0), -1)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(projected_features) * self.noise_std
            projected_features = projected_features + noise
        
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
