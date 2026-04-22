import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TransformerBlock(nn.Module):
    """
    Full transformer block: self-attention + feedforward network.
    Attention routes information between history items.
    FFN transforms that routed information non-linearly.
    Both are needed — attention alone is purely linear.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual + norm
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))

        # FFN with residual + norm
        x = self.norm2(x + self.ffn(x))

        return x


class UserTower(nn.Module):
    """
    User tower:
    - Positional embeddings over watch history sequence
    - Stack of transformer blocks (attention + FFN)
    - Mean pool → concat scalar features → MLP → L2 normalized embedding
    """

    def __init__(
        self,
        history_embed_dim: int,
        scalar_feature_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        max_history_len: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()

        self.history_embed_dim = history_embed_dim
        self.max_history_len = max_history_len

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_history_len, history_embed_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(history_embed_dim, num_attention_heads, dropout)
            for _ in range(num_attention_layers)
        ])

        self.history_norm = nn.LayerNorm(history_embed_dim)

        # MLP
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
            (batch, embedding_dim) L2-normalized user embedding
        """

        batch_size, seq_len, _ = history_embeddings.shape

        # Add positional embeddings
        positions = torch.arange(seq_len, device=history_embeddings.device)
        x = history_embeddings + self.pos_embedding(positions).unsqueeze(0)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        x = self.history_norm(x)

        # Mean pooling
        history_repr = x.mean(dim=1)

        # Concatenate + MLP
        combined = torch.cat([history_repr, scalar_features], dim=-1)
        embedding = self.mlp(combined)

        return F.normalize(embedding, p=2, dim=-1)