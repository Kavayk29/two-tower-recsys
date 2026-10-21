# Context-Aware Two-Tower Recommendation System

[![tests](https://github.com/Kavayk29/two-tower-recsys/actions/workflows/tests.yml/badge.svg)](https://github.com/Kavayk29/two-tower-recsys/actions/workflows/tests.yml)

A candidate-retrieval system in PyTorch on **MovieLens-1M**: a Transformer-based user tower that models each user's viewing sequence, a multi-modal item tower, in-batch contrastive training, FAISS approximate-nearest-neighbour retrieval, and a FastAPI serving layer — tracked with MLflow and covered by CI tests.

## Results

| Metric (full-catalog ranking, held-out validation interactions) | Value |
|---|---|
| **Recall@50** | **0.60** |

Also tracked per epoch in MLflow: training loss, Recall@10, NDCG@10/50, HitRate@k and MRR. `run_ablation.py` compares the full model against an item tower without text embeddings under the same epoch budget.

## Architecture

```
 user history (last 50 items)          item metadata
 ┌──────────────────────────┐          ┌──────────────────────────────┐
 │ MiniLM title embeddings  │          │ genres (multi-hot)           │
 │ + positional embeddings  │          │ MiniLM title embedding (384) │
 │ 3 × Transformer block    │          │ year / popularity            │
 │ (4 heads, padding mask)  │          └──────────────┬───────────────┘
 │ masked mean-pool         │                         │
 │ + user scalars (demogr., │                    MLP 512→256
 │   genre affinity, recency)│                        │
 └────────────┬─────────────┘                         │
         MLP 512→256                                  │
              │            L2-normalised 256-d        │
              └──────────► dot product ◄──────────────┘
                     in-batch softmax (τ = 0.1, logQ correction)
```

- **User tower** — sequence of the user's last 50 positively-rated movies (title embeddings from `all-MiniLM-L6-v2` + learned positions), 3 bidirectional self-attention blocks with key-padding masks, masked mean pooling, concatenated with demographic and behavioural scalar features.
- **Item tower** — genres, sentence-transformer title embeddings and year/popularity features through an MLP.
- **Training** — in-batch sampled softmax with temperature 0.1 and log-Q popularity correction; AdamW, 5-epoch linear warmup then cosine annealing.
- **Retrieval** — item embeddings indexed in FAISS (IVF, `nlist=50`, `nprobe=10`); top-500 candidates per query.
- **Serving** — FastAPI loads the model, index and feature tables once at startup.

## Leakage-safe data pipeline

- Ratings ≥ 4 treated as positives; users with < 5 interactions dropped.
- **Per-user temporal split** (last 10% → test, previous 10% → validation), so the model is always evaluated on each user's *future* interactions.
- **Causal training histories** — every training example's history and behavioural features use only interactions strictly before the target item. `tests/test_dataset.py` is a regression test for this.
- `check_train_fit.py` runs a leave-one-out probe on training users with the same causal construction, to compare train vs. validation recall fairly.

## Project structure

```
src/
├── data/         download.py, preprocess.py (temporal split), features.py (user/item features, MiniLM embeddings)
├── models/       user_tower.py, item_tower.py, two_tower.py
├── training/     dataset.py (causal histories), trainer.py (MLflow), evaluate.py (Recall/NDCG/HitRate/MRR)
├── retrieval/    faiss_index.py
├── serving/      api.py (FastAPI)
└── utils.py
build_index.py      encode the catalog and build the FAISS index
run_ablation.py     with/without text-embedding ablation
check_train_fit.py  train-set leave-one-out diagnostic
tests/              model shape, dataset leakage and API tests (run in GitHub Actions)
configs/config.yaml all hyperparameters
```

## Run it

```bash
pip install -r requirements.txt

python -m src.data.download          # MovieLens-1M
python -m src.data.preprocess        # filtering + temporal split
python -m src.data.features          # user / item features and embeddings
python -m src.training.trainer       # train with MLflow logging
python build_index.py                # FAISS index from the trained item tower

uvicorn src.serving.api:app --port 8000
curl http://localhost:8000/recommend/42?k=10
```

Or with Docker: `docker build -t two-tower . && docker run -p 8000:8000 two-tower` (expects trained artifacts in the image).

View experiments with `mlflow ui` (tracking store: `sqlite:///mlflow.db`). Tests: `pytest tests/ -v`.

To train on a Kaggle GPU, import `notebooks/kaggle_runner.ipynb` (Internet on), which clones this repo and runs the full pipeline.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness / readiness |
| `GET /recommend/{user_id}?k=10` | top-k movie IDs for a known user |

## Tech stack

PyTorch · sentence-transformers · FAISS · FastAPI · MLflow · pandas · pytest · GitHub Actions · Docker
