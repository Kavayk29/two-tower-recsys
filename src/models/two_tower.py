
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.models.user_tower import UserTower
from src.models.item_tower import ItemTower


class TwoTowerModel(nn.Module):

    def __init__(
        self,
        history_embed_dim: int,
        scalar_feature_dim: int,
        item_input_dim: int,
        user_hidden_dims: List[int],
        item_hidden_dims: List[int],
        embedding_dim: int,
        num_attention_heads: int = 4,
        num_attention_layers: int = 3,
        max_history_len: int = 50,
        dropout: float = 0.2,
        temperature: float = 0.1,
        logq_correction: bool = False
    ):
        super().__init__()

        self.temperature = temperature
        self.logq_correction = logq_correction
        self.embedding_dim = embedding_dim

        self.user_tower = UserTower(
            history_embed_dim=history_embed_dim,
            scalar_feature_dim=scalar_feature_dim,
            hidden_dims=user_hidden_dims,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            max_history_len=max_history_len,
            dropout=dropout
        )

        self.item_tower = ItemTower(
            input_dim=item_input_dim,
            hidden_dims=item_hidden_dims,  # now uses its own dims
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
        item_features: torch.Tensor,
        item_sampling_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        user_embeddings = self.user_tower(history_embeddings, scalar_features)
        item_embeddings = self.item_tower(item_features)

        logits = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature

        if self.logq_correction and item_sampling_probs is not None:
            log_probs = torch.log(item_sampling_probs.clamp(min=1e-9))
            logits = logits - log_probs.unsqueeze(0)

        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def get_user_embedding(self, history_embeddings, scalar_features):
        with torch.no_grad():
            return self.user_tower(history_embeddings, scalar_features)

    def get_item_embedding(self, item_features):
        with torch.no_grad():
            return self.item_tower(item_features)