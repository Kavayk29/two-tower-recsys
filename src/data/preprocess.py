import torch
import pandas as pd
import json

def load_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="::", engine="python", header=None, names=["user_id", "item_id", "rating", "timestamp"])
    return df

def filter_data(df: pd.DataFrame, min_rating: float, min_interactions: int):
    df = df[df["rating"] >= min_rating].copy()
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["user_id"].isin(valid_users)]
    return df

def sort_by_time(df: pd.DataFrame):
    return df.sort_values(["user_id", "timestamp"])

def split_by_user_time(df, val_frac=0.1, test_frac=0.1):
    train_rows = []
    val_rows = []
    test_rows = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        n = len(user_df)
        test_size = int(n * test_frac)
        val_size = int(n * val_frac)
        train_end = n - val_size - test_size
        val_end = n - test_size

        train_rows.append(user_df.iloc[:train_end])
        val_rows.append(user_df.iloc[train_end:val_end])
        test_rows.append(user_df.iloc[val_end:])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)
    test_df = pd.concat(test_rows)

    return train_df, val_df, test_df

def build_sequences(df, max_history_len=50):
    sequences = []
    for user_id, user_df in df.groupby("user_id"):
        items = user_df["item_id"].tolist()
        for i in range(1, len(items)):
            history = items[max(0, i - max_history_len):i]
            target = items[i]
            sequences.append({
                "user_id": int(user_id),
                "history": [int(x) for x in history],
                "target": int(target)
            })
    return sequences

def build_eval_data(train_df, eval_df, max_history_len=50):
    eval_data = []
    train_histories = train_df.groupby("user_id")["item_id"].apply(list)
    for user_id, user_df in eval_df.groupby("user_id"):
        if user_id not in train_histories:
            continue
        history = train_histories[user_id][-max_history_len:]
        target = user_df["item_id"].iloc[0]
        eval_data.append({
            "user_id": int(user_id),
            "history": [int(x) for x in history],
            "target": int(target)
        })
    return eval_data
