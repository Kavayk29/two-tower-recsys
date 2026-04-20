import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES = ["ratings.dat", "users.dat", "movies.dat"]
MIN_ZIP_SIZE_MB = 5


def download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name, total=total_size,
        unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def verify_download(zip_path: Path) -> None:
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb < MIN_ZIP_SIZE_MB:
        raise ValueError(f"Downloaded file is {size_mb:.1f}MB — too small, may be corrupt.")
    print(f"Download verified: {size_mb:.1f}MB")


def extract_files(zip_path: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        for target in EXPECTED_FILES:
            zip_member = f"ml-1m/{target}"
            if zip_member not in all_files:
                raise FileNotFoundError(f"{zip_member} not found. Available: {all_files}")
            dest = raw_dir / target
            dest.write_bytes(zf.open(zip_member).read())
            print(f"Extracted: {dest}")


def download_movielens(raw_dir: str = "data/raw") -> None:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    existing = [raw_path / f for f in EXPECTED_FILES]
    if all(p.exists() for p in existing):
        print("MovieLens-1M already downloaded. Skipping.")
        return

    zip_path = raw_path / "ml-1m.zip"
    print(f"Downloading MovieLens-1M from {MOVIELENS_1M_URL}")
    download_file(MOVIELENS_1M_URL, zip_path)
    verify_download(zip_path)
    extract_files(zip_path, raw_path)
    zip_path.unlink()
    print("Download complete.")


if __name__ == "__main__":
    download_movielens()