"""One original case, standard TileLang API, paired DSP timers."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def stats(t, w):
    if any(type(x) is not int or x <= 0 for x in t):
        raise RuntimeError("Invalid ticks")
    v = [x / 19.2 for x in t[w:]]
    return dict(warmup_ticks=t[:w], sample_ticks=t[w:], sample_us=v, median_us=statistics.median(v), min_us=min(v), max_us=max(v), n=len(v))


def execute(a, case):
    import numpy as np
    import torch
    import tilelang

    if (os.environ.get("TILELANG_HEXAGON_HLOS"), os.environ.get("HEXAGON_DSP_ARCH")) != ("linux", "v73"):
        raise RuntimeError("Load environment")
    repo = Path(os.environ["TILELANG_REPO"])
    if Path(tilelang.__file__).resolve().parent != repo / "tilelang":
        raise RuntimeError("Wrong TileLang import")
    from data_files import verify_case

    verify_case(a.data_dir, a.case, a.seed)
    sys.path.insert(0, str(repo / "examples/hexagon"))
    from frozen_ops import input_layout, data_prefix
    from timed_ops import make_timed
    from dsp_power import decode_power

    s = case["spec"]
    m, n = s["m"], s["n"]
    data = a.data_dir / case["group"]
    prefix = data_prefix(case, a.seed)
    paths = []
    inputs = []
    for _name, suffix, shape, dtype in input_layout(case):
        p = data / (prefix + suffix)
        v = np.fromfile(p, dtype="<f2" if dtype == "float16" else "<f4")
        if v.size != int(np.prod(shape)) or not np.isfinite(v).all():
            raise RuntimeError("Invalid input")
        paths.append(p)
        inputs.append(torch.from_numpy(v.reshape(shape).copy()))
    refp = data / (prefix + "_ref.bin")
    ref = np.fromfile(refp, dtype="<f4").astype(np.float64)
    if ref.size != m * n or not np.isfinite(ref).all():
        raise RuntimeError("Invalid reference")
    paths.append(refp)
    func, original, source = make_timed(case, a.warmups, a.samples)
    (a.out / "target_op.cc").write_text(original)
    (a.out / "timer_source.cc").write_text(source)
    print("COMPILE", a.case, json.dumps(s), flush=True)
    index = len(inputs)
    kernel = tilelang.compile(func, out_idx=[index, index + 1, index + 2], target="hexagon")
    (a.out / "timed_op.cc").write_text(kernel.get_kernel_source())
    print("RUN standard TileLang; original config; fixed input publication; paired timers", flush=True)
    output, raw, status = kernel(*inputs)
    words = status.tolist()
    power = decode_power(words, "max")
    save(a.out / "power.json", power)
    if words[0] != 0 or any(x != 0 for x in [*power["request_rc"].values(), *power["cleanup_rc"].values()]):
        raise RuntimeError("DSP or power/cleanup failure: " + str(words))
    actual = output.double().numpy().reshape(-1)
    if actual.shape != ref.shape or not np.isfinite(actual).all():
        raise RuntimeError("Invalid output")
    delta = actual - ref
    nmse = float(np.sum(delta * delta) / max(float(np.sum(actual * actual)), 1e-30))
    limit = 5e-4 if case["group"] == "gemm" else 1e-7
    if not np.isfinite(nmse) or nmse >= limit:
        raise RuntimeError("Accuracy failed " + str(nmse))
    total = a.warmups + a.samples
    t = raw.tolist()
    if len(t) != total * 2 or any(k > f for k, f in zip(t[:total], t[total:])):
        raise RuntimeError("Invalid paired timers")
    metrics = {"dsp_kernel_us": stats(t[:total], a.warmups), "dsp_full_us": stats(t[total:], a.warmups)}
    result = dict(
        schema="original19-readonly/v1",
        case_id=a.case,
        group=case["group"],
        spec=s,
        seed=a.seed,
        warmups=a.warmups,
        samples=a.samples,
        passed=True,
        nmse=nmse,
        max_abs=float(np.max(np.abs(delta))),
        limit=limit,
        metrics=metrics,
        power=power,
        qtimer_hz=19200000,
        input_cache_policy="distinct fixed read-only inputs; explicit publication before timings; no repeated input DMA cleans",
        output_cache_policy="native hex_l2flush policy after every warmup/sample",
        scope={
            "kernel": "generated full operator entry: internal DMA, conversions/packing, compute, output writes, synchronization",
            "full": "same entry plus output publication and nested timer reads; excludes GGML dispatcher preparation",
        },
        submission_policy="one standard TileLang call with internal repetition; differs from GGML graph submissions",
        fully_protocol_equivalent=False,
        original_frozen_spec_unchanged=True,
        inputs={p.name: sha(p) for p in paths},
        sources={
            str(p): sha(p)
            for p in [
                Path(__file__),
                ROOT / "tools/timed_ops.py",
                ROOT / "tools/dsp_power.py",
                ROOT / "tools/frozen_ops.py",
                ROOT / "configs/cases.json",
                repo / "examples/hexagon/example_demo_ops.py",
            ]
        },
    )
    output.numpy().astype("<f4").tofile(a.out / "output.bin")
    save(a.out / "results.json", result)
    save(a.out / "status.json", dict(state="complete", passed=True))
    print("PASS", a.case, "NMSE", nmse, flush=True)
    for key, z in metrics.items():
        print(key, f"median={z['median_us']:.6f} us", flush=True)
    print("RESULTS", a.out / "results.json", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=("run", "bench"), default="bench")
    p.add_argument("--data-dir", type=Path, default=os.environ.get("BENCH19_DATA_DIR"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--samples", type=int, default=128)
    a = p.parse_args()
    cases = json.loads((ROOT / "configs/cases.json").read_text())
    if a.mode == "run":
        a.warmups, a.samples = 0, 1
    if a.case not in cases or not (a.warmups >= 0 and a.samples >= 1 and a.warmups + a.samples <= 4096):
        p.error("Invalid case/counts")
    if not a.data_dir:
        p.error("Specify --data-dir or BENCH19_DATA_DIR (map/ and gemm/ input folders)")
    a.data_dir = a.data_dir.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / "status.json", dict(state="started"))
    try:
        execute(a, cases[a.case])
        return 0
    except Exception as e:
        save(a.out / "status.json", dict(state="failed", error=str(e)))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
