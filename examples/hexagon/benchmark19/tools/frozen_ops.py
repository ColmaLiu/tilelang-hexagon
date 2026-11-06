"""Frozen 19-case definitions and DSP timing through standard TileLang APIs.

This module never creates an ARM agent, FastRPC session or device transport.
TileLang imports are lazy so listings and offline checks need no device/SDK.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def configs():
    return json.loads((ROOT.parent / "configs/cases.json").read_text())


def input_layout(case):
    s = case["spec"]
    m, n = s["m"], s["n"]
    if case["group"] == "gemm":
        return [("A", "_A.bin", (m, s["k"]), "float32"), ("W", "_W.bin", (n, s["k"]), "float16")]
    inputs = [("X", "_src0.bin", (m, n), "float32")]
    if s["kind"] in ("add", "mul"):
        inputs.append(("Y", "_src1.bin", (m, n), "float32"))
    return inputs


def data_prefix(case, seed):
    s = case["spec"]
    name = f"{s['m']}_{s['n']}_{s['k']}" if case["group"] == "gemm" else s["name"]
    return name + f"_s{seed}"


def make_operator(case, ops):
    s = case["spec"]
    if case["group"] == "gemm":
        return ops.make_gemm_f16_io(s["m"], s["n"], s["k"], s["bn"], s["copy_workers"], s["fused_copy"])
    if s["dma"] is None:
        if s["kind"] == "rmsnorm":
            return ops.make_rmsnorm_fp32(s["m"], s["n"], num_workers=s["workers"])
        return ops.make_elementwise(s["m"], s["n"], s["kind"], num_workers=s["workers"])
    return ops.make_map_tiled(
        s["m"], s["n"], s["kind"], s["tile_rows"], s["group_tiles"], num_workers=s["workers"], dma=s["dma"], dma_store=s["dma_store"]
    )
