import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class HistoryAttention(nn.Module):
    """
    Single multi-head self-attention layer applied over the
    watch history embedding sequence before mean pooling.

    This captures intra-sequence relationships in the user's
    history — which items are most relevant to each other —
    rather than treating all history items equally (mean pool).

    Inspired by SASRec (Self-Attentive Sequential Recommendation).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim) — history embeddings
        Returns:
            (batch_size, embed_dim) — attended and mean-pooled representation
        """
        attended, _ = self.attention(x, x, x)
        # Residual connection + layer norm (standard transformer block)
        out = self.norm(x + self.dropout(attended))
        # Mean pool over sequence dimension
        return out.mean(dim=1)  # (batch_size, embed_dim)


class UserTower(nn.Module):
    """
    User tower with two input pathways:

    Pathway 1 — History pathway:
        Watch history embeddings (seq_len, text_embed_dim)
        → Self-attention (HistoryAttention)
        → 384-dim attended representation

    Pathway 2 — Profile pathway:
        Explicit + behavioural scalar features
        → Passed directly to MLP

    Both pathways are concatenated and fed into the MLP
    which projects down to embedding_dim.
    """

    def __init__(
        self,
        history_embed_dim: int,
        scalar_feature_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        num_attention_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.history_embed_dim = history_embed_dim
        self.scalar_feature_dim = scalar_feature_dim

        # Self-attention over history
        self.history_attention = HistoryAttention(
            embed_dim=history_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # MLP input = attended history + scalar features
        mlp_input_dim = history_embed_dim + scalar_feature_dim

        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        scalar_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: (batch, seq_len, history_embed_dim)
            scalar_features: (batch, scalar_feature_dim)
        Returns:
            (batch, embedding_dim) L2-normalised user embedding
        """
        # Self-attention over history → (batch, history_embed_dim)
        history_repr = self.history_attention(history_embeddings)

        # Concatenate with scalar features
        combined = torch.cat([history_repr, scalar_features], dim=-1)

        # MLP projection
        embedding = self.mlp(combined)
        return F.normalize(embedding, p=2, dim=-1)