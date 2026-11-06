"""Verify the original shared inputs without importing TileLang or accessing a board."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_case(data_dir, case, seed):
    index = json.loads((ROOT / "configs/inputs.json").read_text())
    data_dir = Path(data_dir).resolve()
    entries = index["cases"][case][str(seed)]
    for entry in entries:
        path = (data_dir / entry["path"]).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError("Input path escapes data directory")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Input size mismatch: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {path}")
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args()
    cases = json.loads((ROOT / "configs/cases.json").read_text())
    for case in cases:
        for seed in (1, 2, 3):
            verify_case(a.data_dir, case, seed)
    print("PASS original 19 cases, seeds 1/2/3: input sizes and SHA256")


if __name__ == "__main__":
    main()
