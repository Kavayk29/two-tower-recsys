# Context-Aware Two-Tower Recommendation System

This repository contains a PyTorch-based Two-Tower recommendation system using the MovieLens-1M dataset. 
It features a complex **Self-Attention (Transformer) User Tower** for sequence modeling, a multi-modal Item Tower (processing genres and SentenceTransformer title embeddings), and a FAISS-based retrieval pipeline.

## Running on Kaggle
To run this pipeline completely on Kaggle, follow these steps:

1. **Push to GitHub**:
   Ensure this code is pushed to your public GitHub repository. If it's not already on GitHub, initialize git and push it:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/two-tower-recsys.git
   git push -u origin main
   ```

2. **Open Kaggle**:
   - Go to Kaggle and create a **New Notebook**.
   - Make sure your notebook has "Internet on" in the settings panel.
   - You can also turn on "GPU" for faster training and inference.

3. **Import Notebook**:
   - In Kaggle, go to `File > Import Notebook`.
   - Upload the file `notebooks/kaggle_runner.ipynb` from your local machine.
   - Before running the notebook, edit the `!git clone https://github.com/YOUR_USERNAME/two-tower-recsys.git` line to point to your actual GitHub repository URL.

4. **Run All**:
   - Click "Run All". 
   - The notebook will clone your repository, install the necessary dependencies (`sentence-transformers`, `faiss`, `mlflow`), download and extract the dataset, preprocess everything, and start the training process with MLflow logging.

## MLflow Tracking
This project uses **MLflow** to track metrics (Training Loss, Validation Recall@10, Validation NDCG@10, MRR).
When running locally, you can view your metrics by typing:
```bash
mlflow ui
```
And then navigating to `http://localhost:5000` in your browser.
