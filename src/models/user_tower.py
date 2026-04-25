import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TransformerBlock(nn.Module):
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
        seq_len = x.shape[1]

        # Causal mask: position i cannot attend to position j > i
        # causal_mask = torch.triu(
        #     torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        # ).bool()

        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.ffn(x))
        return x


class UserTower(nn.Module):
    def __init__(
        self,
        history_embed_dim: int,
        scalar_feature_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        num_attention_heads: int = 4,
        num_attention_layers: int = 3,
        max_history_len: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()

        self.history_embed_dim = history_embed_dim
        self.max_history_len = max_history_len

        self.pos_embedding = nn.Embedding(max_history_len, history_embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(history_embed_dim, num_attention_heads, dropout)
            for _ in range(num_attention_layers)
        ])

        self.history_norm = nn.LayerNorm(history_embed_dim)

        # Normalize scalar features before they enter the MLP
        self.scalar_norm = nn.LayerNorm(scalar_feature_dim)

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

    def forward(self, history_embeddings, scalar_features):
        batch_size, seq_len, _ = history_embeddings.shape

        positions = torch.arange(seq_len, device=history_embeddings.device)
        x = history_embeddings + self.pos_embedding(positions).unsqueeze(0)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.history_norm(x)

        # Mean pool over real (non-padding) positions only
        # Left-padded: zero rows = padding, non-zero rows = real movies
        real_mask = (history_embeddings.abs().sum(dim=-1) > 0)  # (batch, seq_len)
        real_mask_f = real_mask.unsqueeze(-1).float()       # (batch, seq_len, 1)
        num_real = real_mask_f.sum(dim=1).clamp(min=1)      # (batch, 1)
        history_repr = (x * real_mask_f).sum(dim=1) / num_real  # (batch, 384)

        scalar_features = self.scalar_norm(scalar_features)
        combined = torch.cat([history_repr, scalar_features], dim=-1)
        embedding = self.mlp(combined)
        return F.normalize(embedding, p=2, dim=-1)