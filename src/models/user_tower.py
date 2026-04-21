import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ItemTower(nn.Module):
    """
    Item tower: MLP over concatenated item features.
    Input: genre multi-hot + text embedding + popularity + era
    Output: 128-dim L2-normalised embedding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(x), p=2, dim=-1)