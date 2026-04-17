import torch
import torch.nn as nn

from src.models.user_tower import UserTower
from src.models.item_tower import ItemTower

class TwoTowerModel(nn.Module):
    def __init__(
        self, 
        num_items, 
        num_users, 
        num_genders, 
        num_ages, 
        num_occupations, 
        num_genres, 
        embedding_dim,
        user_hidden_dims,
        item_hidden_dims,
        text_embed_dim=384,
        transformer_heads=4,
        transformer_layers=2
    ):
        super().__init__()

        # shared embedding
        self.item_embedding = nn.Embedding(
            num_items + 1,
            embedding_dim,
            padding_idx=0
        )

        self.user_tower = UserTower(
            self.item_embedding, 
            num_users, num_genders, num_ages, num_occupations, 
            embedding_dim, user_hidden_dims,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )
        
        self.item_tower = ItemTower(
            self.item_embedding, 
            num_genres, 
            embedding_dim, item_hidden_dims, text_embed_dim
        )

    def forward(self, user_features, item_features):
        """
        user_features: dict containing 'user_id', 'gender', 'age', 'occupation', 'history'
        item_features: dict containing 'item_id', 'title_emb', 'genres'
        """
        user_vec = self.user_tower(
            user_features["user_id"],
            user_features["gender"],
            user_features["age"],
            user_features["occupation"],
            user_features["history"]
        )   # (B, D)
        
        item_vec = self.item_tower(
            item_features["item_id"],
            item_features["title_emb"],
            item_features["genres"]
        )    # (B, D)

        logits = torch.matmul(user_vec, item_vec.T)  # (B, B)

        return logits