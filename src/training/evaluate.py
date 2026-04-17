import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import faiss

def compute_metrics(top_k_items, targets):
    """
    top_k_items: (B, K) numpy array of retrieved item IDs
    targets: (B,) numpy array of ground truth item IDs
    """
    hits = 0
    mrr = 0.0
    ndcg = 0.0
    
    for i in range(len(targets)):
        target = targets[i]
        retrieved = top_k_items[i]
        
        if target in retrieved:
            hits += 1
            rank = np.where(retrieved == target)[0][0] + 1
            mrr += 1.0 / rank
            ndcg += 1.0 / np.log2(rank + 1)
            
    n = len(targets)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    return hits / n, mrr / n, ndcg / n

def evaluate_model(model, val_loader, item_features, title_embeddings, device, k=10):
    model.eval()
    
    # 1. Build FAISS Index of all items
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
    
    index = faiss.IndexFlatIP(item_vectors.shape[1])
    id_index = faiss.IndexIDMap(index)
    id_index.add_with_ids(item_vectors, item_ids)
    
    # 2. Evaluate
    all_targets = []
    all_retrieved = []
    
    with torch.no_grad():
        for u_batch, i_batch in val_loader:
            targets = i_batch["item_id"].numpy() # ground truth
            u_batch = {key: val.to(device) for key, val in u_batch.items()}
            
            vecs = model.user_tower(
                u_batch["user_id"], 
                u_batch["gender"], 
                u_batch["age"], 
                u_batch["occupation"], 
                u_batch["history"]
            )
            vecs_np = vecs.cpu().numpy().astype("float32")
            
            # Retrieve top K
            scores, I = id_index.search(vecs_np, k)
            
            all_retrieved.append(I)
            all_targets.append(targets)
            
    if len(all_retrieved) == 0:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
        
    all_retrieved = np.vstack(all_retrieved)
    all_targets = np.concatenate(all_targets)
    
    recall, mrr, ndcg = compute_metrics(all_retrieved, all_targets)
    
    model.train() # restore train mode
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}
