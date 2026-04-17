import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import json
import yaml
import faiss
import numpy as np

from src.models.two_tower import TwoTowerModel

def load_model_and_data():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("data/processed/user_features.json", "r") as f:
        user_data = json.load(f)
        user_features = user_data["features"]
        user_maps = user_data["maps"]
        
    with open("data/processed/item_features.json", "r") as f:
        item_data = json.load(f)
        item_features = item_data["features"]
        item_maps = item_data["maps"]
        
    with open("data/processed/train.json", "r") as f:
        train_sequences = json.load(f)
        
    title_embeddings = torch.load("data/processed/title_embeddings.pt").to(device)
    
    num_users = 6040
    num_genders = len(user_maps["gender"])
    num_ages = len(user_maps["age"])
    num_occupations = len(user_maps["occupation"])
    
    num_items = 3952
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
        item_hidden_dims=config["model"]["item_hidden_dims"]
    ).to(device)
    
    model.load_state_dict(torch.load("data/processed/model.pt", map_location=device))
    model.eval()
    
    # Build user histories
    user_histories = {}
    for seq in train_sequences:
        uid = seq["user_id"]
        if uid not in user_histories:
            user_histories[uid] = []
        user_histories[uid].append(seq["target"])
        
    for uid in user_histories:
        user_histories[uid] = user_histories[uid][-50:] # max_history length
        
    return model, config, user_features, item_features, user_histories, title_embeddings, device

def build_faiss_index(model, item_features, title_embeddings, device):
    item_ids = []
    item_vectors = []
    
    max_genres = 6
    
    with torch.no_grad():
        for iid_str, feats in item_features.items():
            iid = int(iid_str)
            genres = feats["genres"][:max_genres]
            genres_padded = np.zeros(max_genres, dtype=np.int64)
            genres_padded[:len(genres)] = genres
            
            i_batch = {
                "item_id": torch.tensor([iid]).to(device),
                "title_emb": title_embeddings[iid].unsqueeze(0).to(device),
                "genres": torch.tensor([genres_padded]).to(device)
            }
            
            vec = model.item_tower(i_batch["item_id"], i_batch["title_emb"], i_batch["genres"])
            item_vectors.append(vec.cpu().numpy()[0])
            item_ids.append(iid)
            
    item_vectors = np.array(item_vectors).astype("float32")
    item_ids = np.array(item_ids).astype("int64")
    
    # Inner product index for cosine similarity/dot product
    index = faiss.IndexFlatIP(item_vectors.shape[1])
    
    # Wrap with IDMap
    id_index = faiss.IndexIDMap(index)
    id_index.add_with_ids(item_vectors, item_ids)
    
    return id_index

def recommend(user_id, model, index, user_features, user_histories, device, top_k=5):
    if str(user_id) not in user_features:
        print("User not found.")
        return []
        
    u_feat = user_features[str(user_id)]
    
    hist = user_histories.get(user_id, [])
    hist_padded = np.zeros(50, dtype=np.int64)
    hist_padded[:len(hist)] = hist
    
    u_batch = {
        "user_id": torch.tensor([user_id]).to(device),
        "gender": torch.tensor([u_feat["gender"]]).to(device),
        "age": torch.tensor([u_feat["age"]]).to(device),
        "occupation": torch.tensor([u_feat["occupation"]]).to(device),
        "history": torch.tensor([hist_padded]).to(device)
    }
    
    with torch.no_grad():
        vec = model.user_tower(u_batch["user_id"], u_batch["gender"], u_batch["age"], u_batch["occupation"], u_batch["history"])
        vec_np = vec.cpu().numpy().astype("float32")
        
    scores, I = index.search(vec_np, top_k)
    return I[0], scores[0]

if __name__ == "__main__":
    print("Loading data and model...")
    model, config, user_features, item_features, user_histories, title_embeddings, device = load_model_and_data()
    
    print("Building FAISS index...")
    index = build_faiss_index(model, item_features, title_embeddings, device)
    
    test_user = 1
    print(f"\nRecommendations for User {test_user}:")
    rec_ids, scores = recommend(test_user, model, index, user_features, user_histories, device, top_k=5)
    
    import pandas as pd
    movies = pd.read_csv("data/raw/movies.dat", sep="::", engine="python", header=None, names=["item_id", "title", "genres"], encoding="latin-1")
    movie_dict = dict(zip(movies.item_id, movies.title))
    
    for i, iid in enumerate(rec_ids):
        print(f"{i+1}. {movie_dict.get(iid, 'Unknown')} (Score: {scores[i]:.4f})")
