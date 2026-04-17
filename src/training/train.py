import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import yaml
import mlflow
import numpy as np

from src.models.two_tower import TwoTowerModel
from src.training.evaluate import evaluate_model

class RecSysDataset(Dataset):
    def __init__(self, data_path, user_features_path, item_features_path, title_embeddings_path, max_history=50, max_genres=6):
        with open(data_path, "r") as f:
            self.data = json.load(f)
            
        with open(user_features_path, "r") as f:
            self.user_features = json.load(f)["features"]
            
        with open(item_features_path, "r") as f:
            self.item_features = json.load(f)["features"]
            
        self.title_embeddings = torch.load(title_embeddings_path)
        
        self.max_history = max_history
        self.max_genres = max_genres
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        user_id = row["user_id"]
        target = row["target"]
        history = row["history"]
        
        u_feat = self.user_features[str(user_id)]
        i_feat = self.item_features[str(target)]
        
        # pad history
        hist = history[-self.max_history:]
        hist_padded = np.zeros(self.max_history, dtype=np.int64)
        hist_padded[:len(hist)] = hist
        
        # pad genres
        genres = i_feat["genres"][:self.max_genres]
        genres_padded = np.zeros(self.max_genres, dtype=np.int64)
        genres_padded[:len(genres)] = genres
        
        return {
            "user_features": {
                "user_id": user_id,
                "gender": u_feat["gender"],
                "age": u_feat["age"],
                "occupation": u_feat["occupation"],
                "history": hist_padded
            },
            "item_features": {
                "item_id": target,
                "title_emb": self.title_embeddings[target],
                "genres": genres_padded
            }
        }

def collate_fn(batch):
    user_features = {
        "user_id": torch.tensor([b["user_features"]["user_id"] for b in batch]),
        "gender": torch.tensor([b["user_features"]["gender"] for b in batch]),
        "age": torch.tensor([b["user_features"]["age"] for b in batch]),
        "occupation": torch.tensor([b["user_features"]["occupation"] for b in batch]),
        "history": torch.tensor(np.stack([b["user_features"]["history"] for b in batch]))
    }
    
    item_features = {
        "item_id": torch.tensor([b["item_features"]["item_id"] for b in batch]),
        "title_emb": torch.stack([b["item_features"]["title_emb"] for b in batch]),
        "genres": torch.tensor(np.stack([b["item_features"]["genres"] for b in batch]))
    }
    
    return user_features, item_features

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = RecSysDataset(
        "data/processed/train.json", 
        "data/processed/user_features.json", 
        "data/processed/item_features.json", 
        "data/processed/title_embeddings.pt"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    
    val_dataset = RecSysDataset(
        "data/processed/val.json", 
        "data/processed/user_features.json", 
        "data/processed/item_features.json", 
        "data/processed/title_embeddings.pt"
    )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Get vocabulary sizes from maps
    with open("data/processed/user_features.json", "r") as f:
        maps = json.load(f)["maps"]
        num_users = 6040 # ML-1M max user id
        num_genders = len(maps["gender"])
        num_ages = len(maps["age"])
        num_occupations = len(maps["occupation"])
        
    with open("data/processed/item_features.json", "r") as f:
        item_maps = json.load(f)["maps"]
        num_items = 3952 # ML-1M max item id
        num_genres = len(item_maps["genre"])
        
    model = TwoTowerModel(
        num_items=num_items,
        num_users=num_users,
        num_genders=num_genders,
        num_ages=num_ages,
        num_occupations=num_occupations,
        num_genres=num_genres,
        embedding_dim=config["model"]["embedding_dim"],
        user_hidden_dims=config["model"]["user_hidden_dims"],
        item_hidden_dims=config["model"]["item_hidden_dims"],
        transformer_heads=config["model"].get("transformer_heads", 4),
        transformer_layers=config["model"].get("transformer_layers", 2)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    temperature = config["model"]["temperature"]
    
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "embedding_dim": config["model"]["embedding_dim"],
            "user_hidden_dims": config["model"]["user_hidden_dims"],
            "item_hidden_dims": config["model"]["item_hidden_dims"],
            "transformer_heads": config["model"].get("transformer_heads", 4),
            "transformer_layers": config["model"].get("transformer_layers", 2),
            "temperature": temperature
        })
        
        for epoch in range(config["training"]["epochs"]):
            model.train()
            total_loss = 0
            for i, (u_batch, i_batch) in enumerate(train_loader):
                # move to device
                u_batch = {k: v.to(device) for k, v in u_batch.items()}
                i_batch = {k: v.to(device) for k, v in i_batch.items()}
                
                optimizer.zero_grad()
                
                logits = model(u_batch, i_batch)
                logits = logits / temperature
                
                # In batch negative sampling: target is the diagonal
                labels = torch.arange(logits.size(0)).to(device)
                loss = nn.functional.cross_entropy(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % 50 == 0:
                    print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                    
            avg_loss = total_loss/len(train_loader)
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Evaluate
            print("Evaluating on validation set...")
            metrics = evaluate_model(model, val_loader, train_dataset.item_features, train_dataset.title_embeddings, device, k=10)
            print(f"Val Metrics - Recall@10: {metrics['recall']:.4f}, NDCG@10: {metrics['ndcg']:.4f}, MRR@10: {metrics['mrr']:.4f}")
            
            mlflow.log_metrics({
                "val_recall_10": metrics["recall"],
                "val_ndcg_10": metrics["ndcg"],
                "val_mrr_10": metrics["mrr"]
            }, step=epoch)
            
        # Save model
        torch.save(model.state_dict(), "data/processed/model.pt")
        mlflow.log_artifact("data/processed/model.pt")
        print("Training finished.")

if __name__ == "__main__":
    main()
