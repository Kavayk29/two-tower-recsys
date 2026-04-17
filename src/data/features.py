import torch
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

def process_user_features(users_path, out_path):
    users = pd.read_csv(users_path, sep="::", engine="python", header=None, names=["user_id", "gender", "age", "occupation", "zip"])
    
    # Mappings
    gender_map = {str(g): i for i, g in enumerate(users["gender"].unique())}
    age_map = {int(a): i for i, a in enumerate(users["age"].unique())}
    occ_map = {int(o): i for i, o in enumerate(users["occupation"].unique())}
    
    user_features = {}
    for _, row in users.iterrows():
        user_features[int(row["user_id"])] = {
            "gender": gender_map[str(row["gender"])],
            "age": age_map[int(row["age"])],
            "occupation": occ_map[int(row["occupation"])]
        }
        
    with open(out_path, "w") as f:
        json.dump({
            "features": user_features,
            "maps": {
                "gender": gender_map,
                "age": age_map,
                "occupation": occ_map
            }
        }, f)

def process_item_features(movies_path, out_path_json, out_path_pt, embed_model_name):
    movies = pd.read_csv(movies_path, sep="::", engine="python", header=None, names=["item_id", "title", "genres"], encoding="latin-1")
    
    # process genres
    all_genres = set()
    for g_str in movies["genres"]:
        all_genres.update(g_str.split("|"))
    genre_map = {g: i for i, g in enumerate(sorted(all_genres))}
    
    item_features = {}
    titles = []
    item_ids = []
    
    for _, row in movies.iterrows():
        g_list = str(row["genres"]).split("|")
        g_indices = [genre_map[g] for g in g_list]
        item_features[int(row["item_id"])] = {
            "genres": g_indices
        }
        titles.append(str(row["title"]))
        item_ids.append(int(row["item_id"]))
        
    # generate text embeddings
    print("Generating title embeddings...")
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
    
    # Save a tensor dict
    # embeddings: (num_items, dim). Let's map item_id to embedding using a zero-padded tensor
    max_id = max(item_ids)
    dim = embeddings.shape[1]
    item_embeddings = torch.zeros(max_id + 1, dim)
    for i, iid in enumerate(item_ids):
        item_embeddings[iid] = embeddings[i].cpu()
        
    torch.save(item_embeddings, out_path_pt)
    
    with open(out_path_json, "w") as f:
        json.dump({
            "features": item_features,
            "maps": {
                "genre": genre_map
            }
        }, f)
