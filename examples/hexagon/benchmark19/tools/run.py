"""Run one or all original cases serially through the standard TileLang runtime."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = json.loads((ROOT / "configs/cases.json").read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", nargs="?", choices=["all", *cases], default="all")
    p.add_argument("mode", nargs="?", choices=["run", "bench"], default="bench")
    p.add_argument("--list", action="store_true")
    p.add_argument("--data-dir", type=Path, default=os.environ.get("BENCH19_DATA_DIR"))
    p.add_argument("--output-dir", type=Path, default=os.environ.get("BENCH19_OUTPUT_DIR", "/output/benchmark19"))
    p.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--samples", type=int, default=128)
    a = p.parse_args()
    if a.list:
        print("\n".join(cases))
        return 0
    if not a.data_dir:
        p.error("Set BENCH19_DATA_DIR or pass --data-dir")
    if a.warmups < 0 or a.samples < 1 or a.warmups + a.samples > 4096:
        p.error("Invalid warmup/sample counts")
    from data_files import verify_case

    selected = list(cases) if a.case == "all" else [a.case]
    for case in selected:
        verify_case(a.data_dir, case, a.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    out = a.output_dir.resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    print("OUTPUT", out, flush=True)
    # Cooperating processes for this local user share the same file across copies.
    lock_path = Path(tempfile.gettempdir()) / f"tilelang-benchmark19-{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another benchmark19 runner for this user is active") from None
        records = []
        for case in selected:
            command = [
                sys.executable,
                str(ROOT / "tools/run_tilelang.py"),
                "--case",
                case,
                "--mode",
                a.mode,
                "--data-dir",
                str(a.data_dir.resolve()),
                "--out",
                str(out / case),
                "--seed",
                str(a.seed),
                "--warmups",
                str(a.warmups),
                "--samples",
                str(a.samples),
            ]
            print("CASE", case, flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode:
                (out / "status.json").write_text(
                    json.dumps(
                        {
                            "state": "failed",
                            "case": case,
                            "completed": len(records),
                            "planned": len(selected),
                            "returncode": result.returncode,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return result.returncode
            records.append(json.loads((out / case / "results.json").read_text()))
            (out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        (out / "status.json").write_text(
            json.dumps(
                {
                    "state": "complete",
                    "completed": len(records),
                    "planned": len(selected),
                },
                indent=2,
            )
            + "\n"
        )
    print("RESULTS", out / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
