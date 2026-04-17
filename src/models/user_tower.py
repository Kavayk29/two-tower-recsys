import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]

class UserTower(nn.Module):
    def __init__(self, item_embedding, num_users, num_genders, num_ages, num_occupations, embedding_dim, hidden_dims, transformer_heads=4, transformer_layers=2):
        super().__init__()

        self.item_embedding = item_embedding

        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.gender_embedding = nn.Embedding(num_genders + 1, 16, padding_idx=0)
        self.age_embedding = nn.Embedding(num_ages + 1, 16, padding_idx=0)
        self.occ_embedding = nn.Embedding(num_occupations + 1, 16, padding_idx=0)
        
        # Transformer for sequence history modeling
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=200)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=transformer_heads, 
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        input_dim = embedding_dim * 2 + 48

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)

    def forward(self, user_id, gender, age, occupation, history):
        """
        history: (B, max_len)
        """
        # (B, max_len, D)
        embedded_hist = self.item_embedding(history)
        
        # Add positional encoding
        embedded_hist = self.pos_encoder(embedded_hist)

        # padding mask for transformer (True means IGNORE)
        src_key_padding_mask = (history == 0) # (B, max_len)
        
        # pass through transformer
        # output is (B, max_len, D)
        transformer_out = self.transformer(embedded_hist, src_key_padding_mask=src_key_padding_mask)

        # mask padding (0s) for pooling
        mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, max_len, 1)
        # Avoid NaN when all history is padded (e.g. cold start user)
        transformer_out = transformer_out.masked_fill(mask == 0, 0.0)

        # mean pooling over the sequence
        sum_embeddings = transformer_out.sum(dim=1)  # (B, D)
        lengths = mask.sum(dim=1)  # (B, 1)
        hist_vec = sum_embeddings / (lengths + 1e-8)

        u_vec = self.user_embedding(user_id)
        g_vec = self.gender_embedding(gender)
        a_vec = self.age_embedding(age)
        o_vec = self.occ_embedding(occupation)

        concat_vec = torch.cat([u_vec, g_vec, a_vec, o_vec, hist_vec], dim=-1)

        return self.dnn(concat_vec)