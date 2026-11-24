"""Stage 1a (M1): download the WMT26 Indic MT 2026 release from the official Google Drive links.

Links are those published on https://www2.statmt.org/wmt26/indic-mt-task.html .
The two test-set zips are password protected (released only to registered teams); we
download them for completeness / checksum logging but cannot open them.

Usage:  python -m src.data.download
"""
from __future__ import annotations

import hashlib
import json

from src.utils import RAW

TRAIN_FOLDER = "https://drive.google.com/drive/folders/1TsaINvXwb5GbiEbEkj2OgcK-1wfrWESG"
TEST_2026 = "1naayXMtVp01s1km6-w7mfPK3oY69DV2g"
TEST_2025_GOLD = "1C2slIg368xamg9_-AnuwN5rFpVeBO1ct"

ORIG = RAW / "_original"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import gdown

    ORIG.mkdir(parents=True, exist_ok=True)
    train_dir = ORIG / "training_2026"
    if not train_dir.exists():
        gdown.download_folder(TRAIN_FOLDER, output=str(train_dir), quiet=False)
    for fid, name in [(TEST_2026, "test2026.zip"), (TEST_2025_GOLD, "test2025.zip")]:
        if not (ORIG / name).exists():
            gdown.download(id=fid, output=str(ORIG / name), quiet=False)

    sums = {str(p.relative_to(ORIG)).replace("\\", "/"): sha256(p)
            for p in sorted(ORIG.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    with open(ORIG / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(sums, f, indent=2, ensure_ascii=False)
    print(f"Downloaded {len(sums)} files; checksums -> {ORIG / 'checksums.json'}")


if __name__ == "__main__":
    main()
