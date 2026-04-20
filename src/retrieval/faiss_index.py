import faiss
import numpy as np
import torch
import pandas as pd
from pathlib import Path


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "IVF",
    nlist: int = 50
) -> faiss.Index:
    dim = embeddings.shape[1]
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "IVF":
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )
        print(f"Training IVF index (nlist={nlist})...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index, movie_ids: np.ndarray, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_dir / "item_index.faiss"))
    np.save(save_dir / "movie_ids.npy", movie_ids)
    print(f"Index saved to {save_dir}/")


def load_index(save_dir: Path):
    index = faiss.read_index(str(save_dir / "item_index.faiss"))
    movie_ids = np.load(save_dir / "movie_ids.npy")
    print(f"Index loaded: {index.ntotal} vectors")
    return index, movie_ids


def build_and_save_index(
    model,
    item_features: pd.DataFrame,
    save_dir: Path,
    device: torch.device,
    index_type: str = "IVF",
    nlist: int = 50,
    batch_size: int = 256
) -> None:
    item_feat_cols = [
        c for c in item_features.columns
        if c not in ["movieId", "year"]
    ]
    movie_ids = item_features["movieId"].values
    feat_matrix = item_features[item_feat_cols].values.astype(np.float32)

    model.eval()
    all_embeddings = []

    print(f"Encoding {len(movie_ids):,} items...")
    for start in range(0, len(feat_matrix), batch_size):
        batch = torch.tensor(
            feat_matrix[start:start + batch_size], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            emb = model.item_tower(batch).cpu().numpy()
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)
    index = build_faiss_index(all_embeddings, index_type, nlist)
    save_index(index, movie_ids, save_dir)


def retrieve_top_k(
    index,
    movie_ids: np.ndarray,
    user_embedding: np.ndarray,
    k: int = 500
) -> list:
    query = np.ascontiguousarray(
        user_embedding.reshape(1, -1).astype(np.float32)
    )
    _, indices = index.search(query, k)
    return movie_ids[indices[0]].tolist()