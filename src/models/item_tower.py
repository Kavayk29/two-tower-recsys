import torch
import torch.nn as nn

class ItemTower(nn.Module):
    def __init__(self, item_embedding, num_genres, embedding_dim, hidden_dims, text_embed_dim=384):
        super().__init__()

        self.item_embedding = item_embedding
        self.genre_embedding = nn.Embedding(num_genres + 1, 16, padding_idx=0)

        # Assuming text_embed_dim is 384 for sentence-transformers/all-MiniLM-L6-v2
        input_dim = embedding_dim + text_embed_dim + 16

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)

    def forward(self, item_id, title_emb, genres):
        """
        item_id: (B,)
        title_emb: (B, 384)
        genres: (B, max_genres)
        """
        # Genre mean pooling
        g_emb = self.genre_embedding(genres) # (B, max_genres, 16)
        mask = (genres != 0).float().unsqueeze(-1)
        g_emb = g_emb * mask
        g_sum = g_emb.sum(dim=1)
        g_lengths = mask.sum(dim=1)
        g_vec = g_sum / (g_lengths + 1e-8)

        # Item ID embedding
        i_vec = self.item_embedding(item_id)

        concat_vec = torch.cat([i_vec, title_emb, g_vec], dim=-1)

        return self.dnn(concat_vec)