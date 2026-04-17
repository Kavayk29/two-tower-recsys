import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES = ["ratings.dat", "movies.dat", "users.dat"]
MIN_ZIP_SIZE_MB = 4


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def verify_download(zip_path: Path) -> None:
    """Verify the downloaded zip is intact and large enough."""
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb < MIN_ZIP_SIZE_MB:
        raise ValueError(
            f"Downloaded file is {size_mb:.1f}MB — expected at least "
            f"{MIN_ZIP_SIZE_MB}MB. Download may be corrupt."
        )
    print(f"Download verified: {size_mb:.1f}MB")


def extract_files(zip_path: Path, raw_dir: Path) -> None:
    """Extract only the files we need from the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()

        for target in EXPECTED_FILES:
            # Files inside zip are under ml-1m/ folder
            zip_member = f"ml-1m/{target}"
            if zip_member not in all_files:
                raise FileNotFoundError(
                    f"{zip_member} not found in zip. "
                    f"Available files: {all_files}"
                )
            # Extract and flatten — save directly to raw_dir
            source = zf.open(zip_member)
            dest = raw_dir / target
            dest.write_bytes(source.read())
            print(f"Extracted: {dest}")


def download_movielens(raw_dir: str = "data/raw") -> None:
    """
    Main entry point. Downloads and extracts MovieLens-1M
    to raw_dir. Skips download if files already exist.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = [raw_path / f for f in EXPECTED_FILES]
    if all(p.exists() for p in existing):
        print("MovieLens-1M already downloaded. Skipping.")
        return

    zip_path = raw_path / "ml-1m.zip"

    print(f"Downloading MovieLens-1M from {MOVIELENS_URL}")
    download_file(MOVIELENS_URL, zip_path)

    verify_download(zip_path)

    print("Extracting files...")
    extract_files(zip_path, raw_path)

    # Remove zip after extraction to save space
    zip_path.unlink()
    print("Zip removed. Download complete.")

    # Print summary
    for f in existing:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f}MB")


if __name__ == "__main__":
    download_movielens()