import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    print(f"Ratings loaded: {len(ratings):,} rows")
    return ratings


def load_users(raw_dir: Path) -> pd.DataFrame:
    """
    Load explicit user profiles.
    Columns: userId, gender, age, occupation, zip_code
    Age is bucketed: 1=under18, 18, 25, 35, 45, 50, 56
    Occupation: 0-20 integer codes
    """
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        dtype={"userId": np.int32, "age": np.int32, "occupation": np.int32}
    )
    # Encode gender: M=1, F=0
    users["gender_encoded"] = (users["gender"] == "M").astype(np.float32)
    # Normalise age bucket (max bucket is 56)
    users["age_norm"] = (users["age"] / 56.0).astype(np.float32)
    # Normalise occupation (21 categories, 0-20)
    users["occupation_norm"] = (users["occupation"] / 20.0).astype(np.float32)
    users = users.drop(columns=["gender", "zip_code"])
    print(f"Users loaded: {len(users):,} rows")
    return users


def load_movies(raw_dir: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
        dtype={"movieId": np.int32}
    )
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies["title_clean"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)", "", regex=True
    ).str.strip()
    print(f"Movies loaded: {len(movies):,} rows")
    return movies


def filter_interactions(
    ratings: pd.DataFrame, min_rating: float, min_interactions: int
) -> pd.DataFrame:
    positive = ratings[ratings["rating"] >= min_rating].copy()
    print(f"After rating filter (>={min_rating}): {len(positive):,} rows")

    user_counts = positive.groupby("userId").size()
    active_users = user_counts[user_counts >= min_interactions].index
    positive = positive[positive["userId"].isin(active_users)].copy()
    print(f"After min_interactions filter: {len(positive):,} rows")
    print(f"Unique users: {positive['userId'].nunique():,}")
    print(f"Unique movies: {positive['movieId'].nunique():,}")
    return positive


def temporal_split(
    interactions: pd.DataFrame, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_rows, val_rows, test_rows = [], [], []
    
    for user_id, group in interactions.groupby("userId"):
        group = group.sort_values("timestamp")
        n = len(group)
        
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_rows.append(group.tail(n_test))
        val_rows.append(group.iloc[-(n_test + n_val):-n_test])
        train_rows.append(group.iloc[:-(n_test + n_val)])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)

    return train, val, test

def cap_samples_per_user(
    interactions: pd.DataFrame, max_samples: int = 50
) -> pd.DataFrame:
    capped = (
        interactions
        .sort_values("timestamp")
        .groupby("userId")
        .tail(max_samples)
    )
    print(f"After capping at {max_samples} samples/user: {len(capped):,} rows")
    return capped.reset_index(drop=True)


def run_preprocessing(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(raw_dir)
    users = load_users(raw_dir)
    movies = load_movies(raw_dir)

    movies.to_parquet(processed_dir / "movies.parquet", index=False)
    users.to_parquet(processed_dir / "users.parquet", index=False)
    print("Saved movies.parquet and users.parquet")

    interactions = filter_interactions(
        ratings,
        min_rating=config["data"]["min_rating"],
        min_interactions=config["data"]["min_interactions"]
    )

    train, val, test = temporal_split(
        interactions,
        val_frac=config["data"]["val_frac"],
        test_frac=config["data"]["test_frac"]
    )

    train = cap_samples_per_user(
        train, max_samples=config["data"]["max_samples_per_user"]
    )

    train.to_parquet(processed_dir / "train_interactions.parquet", index=False)
    val.to_parquet(processed_dir / "val_interactions.parquet", index=False)
    test.to_parquet(processed_dir / "test_interactions.parquet", index=False)

    print(f"\nAll files saved to {processed_dir}/")
    print("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()