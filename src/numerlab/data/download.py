from pathlib import Path

from rootutils import find_root

from numerlab.data.datamodule import NumeraiDataset


def main() -> None:
    """Download Numerai dataset to the data directory.

    This function instantiates a NumeraiDataset which triggers the
    download.
    """
    data_dir: Path = find_root() / "data"

    # Move files to force re-download
    files = [
        data_dir / "numerai.parquet",
        data_dir / "eras.json",
        data_dir / "features.json",
        data_dir / "v5.0" / "validation.parquet",
    ]
    target_dir = data_dir / "old"
    target_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file.exists():
            file.rename(target_dir / file.name)

    # Download dataset
    NumeraiDataset(data_dir, download=True)


if __name__ == "__main__":
    main()
