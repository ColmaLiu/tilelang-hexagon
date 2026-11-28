"""Elementwise maps and K reductions of ordinary and paired integer panels."""

from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel, unpack_panel
from test_tilelang_hexagon_dequant_sim import DEVICE, SIMULATOR, _tool
from test_tilelang_hexagon_hmx_int import _gemm
from test_tilelang_hexagon_vectorize import _source

N, K, G = 128, 2048, 64


def _requant():
    @T.prim_func
    def requant(
        Q: T.Tensor((N, K), "int4"),
        S: T.Tensor((K // G, N), "float16"),
        W: T.Tensor((N, K), "int8"),
        Scale: T.Tensor((N,), "float32"),
        Sums: T.Tensor((N,), "int32"),
        Factors: T.Tensor((K // G, N), "int16"),
    ):
        with T.Kernel(1):
            q = T.alloc_shared((N, K), "int4")
            w = T.alloc_shared((N, K), "int8")
            s = T.alloc_shared((K // G, N), "float16")
            sa = T.alloc_shared((K // G, N), "uint16")
            max_bits = T.alloc_shared((N,), "uint16")
            scale_bits = T.view(s, dtype="uint16")
            max_half = T.view(max_bits, dtype="float16")
            m = T.alloc_shared((N,), "float32")
            reciprocal = T.alloc_shared((N,), "float32")
            f = T.alloc_shared((K // G, N), "int16")
            cs = T.alloc_shared((N,), "int32")
            T.annotate_layout({Q: make_hvx_panel_layout(Q), W: make_hvx_panel_layout(W, paired=True)})
            T.copy(Q, q)
            T.copy(S, s)
            for g, n in T.Parallel(K // G, N):
                # Finite nonnegative binary16 values have the same order as their bits.
                sa[g, n] = scale_bits[g, n] & T.uint16(0x7FFF)
            T.reduce_max(sa, max_bits, dim=0)
            for n in T.Parallel(N):
                m[n] = T.Cast("float32", max_half[n])
            for n in T.Parallel(N):
                reciprocal[n] = T.float32(4064) / T.max(m[n], T.float32(1e-30))
            for g, n in T.Parallel(K // G, N):
                value = T.Cast("float32", s[g, n]) * reciprocal[n]
                f[g, n] = T.Cast("int16", value + T.Select(value < 0, T.float32(-0.5), T.float32(0.5)))
            for n, k in T.Parallel(N, K):
                w[n, k] = T.Cast("int8", (T.Cast("int16", q[n, k]) * f[k // G, n] + T.int16(128)) >> 8)
            T.reduce_sum(w, cs, dim=1)
            for n in T.Parallel(N):
                Scale[n] = m[n] * T.float32(8 / 127)
            T.copy(cs, Sums)
            T.copy(w, W)
            T.copy(f, Factors)

    return requant


def _map(dtype, paired, per_row):
    rows, cols = 64, 128

    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), dtype), B: T.Tensor((cols // 8, rows), "int16"), C: T.Tensor((rows, cols), "int8")):
        with T.Kernel(1):
            T.annotate_layout({A: make_hvx_panel_layout(A), C: make_hvx_panel_layout(C, paired=paired)})
            for r, c in T.Parallel(rows, cols):
                C[r, c] = T.Cast("int8", T.Cast("int16", A[r, c]) + B[0 if per_row else c // 8, r])

    return kernel


def _float_factors():
    @T.prim_func
    def kernel(A: T.Tensor((64, 96), "uint4"), B: T.Tensor((4, 64), "float32"), C: T.Tensor((64, 96), "int8")):
        with T.Kernel(1):
            T.annotate_layout({A: make_hvx_panel_layout(A), C: make_hvx_panel_layout(C, paired=True)})
            for r, c in T.Parallel(64, 96):
                C[r, c] = T.Cast("int8", T.Cast("float32", A[r, c]) + B[c // 24, r])

    return kernel


def _as_float(paired, rows=64, cols=128):
    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), "int8"), B: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1):
            T.annotate_layout({A: make_hvx_panel_layout(A, paired=paired)})
            for r, c in T.Parallel(rows, cols):
                B[r, c] = T.Cast("float32", A[r, c])

    return kernel


def _sum(paired, clear=True, reduce_max=False):
    @T.prim_func
    def kernel(W: T.Tensor((64, 128), "int8"), Initial: T.Tensor((64,), "int32"), Sums: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            w = T.alloc_shared((64, 128), "int8")
            s = T.alloc_shared((64,), "int32")
            T.annotate_layout({W: make_hvx_panel_layout(W, paired=paired)})
            T.copy(W, w)
            T.copy(Initial, s)
            if reduce_max:
                T.reduce_max(w, s, dim=1)
            else:
                T.reduce_sum(w, s, dim=1, clear=clear)
            T.copy(s, Sums)

    return kernel


def _inputs():
    rng = np.random.default_rng(912)
    codes = rng.integers(-8, 8, (N, K), dtype=np.int8)
    scales = (rng.standard_normal((K // G, N)) * 0.02).astype(np.float16)
    scales[:, 0] = 0
    scales[:, 1] = np.float16(-0.0)
    # Subnormal half, maximum finite half, and exact half-integer factors.
    scales[:, 2] = np.resize(np.array([0, 1, 0x3FF, 0x8001], np.uint16).view(np.float16), K // G)
    scales[:, 3] = np.resize(np.array([65504, -32752, 0], np.float16), K // G)
    # The approximate vector reciprocal on both targets moves 91.5 below
    # its tie. For code 7 this also changes the final byte.
    scales[:3, 5] = [1524, 34.3125, -34.3125]
    scales[:3, 4] = [1, -1 / 64, 3 / 64]
    return codes, scales


def _check_requant(codes, scales, weights, scale, sums, factors):
    # Check the actual factor before shifting away its low bits. Two Newton
    # steps in fast math may move a quotient across a nearby half integer.
    s = scales.astype(np.float64)
    maximum = np.abs(s).max(axis=0)
    ideal = s * (4064 / np.maximum(maximum, 1e-30))
    expected = np.copysign(np.floor(np.abs(ideal) + 0.5), ideal)
    error = np.abs(factors - expected)
    assert error.max() <= 1
    changed = error != 0
    assert changed.mean() < 0.005
    distance_to_tie = np.abs(np.abs(ideal) % 1 - 0.5)
    assert np.all(distance_to_tie[changed] < 0.04)

    # The fixed-point map, reduction and per-channel scale each have an
    # independent exact check, including rows whose scale is zero.
    expected_weights = (codes.astype(np.int32) * factors.T.repeat(G, axis=1) + 128) >> 8
    np.testing.assert_array_equal(weights, expected_weights)
    assert np.max(np.abs(weights.astype(np.int32))) <= 127
    np.testing.assert_array_equal(sums, weights.sum(axis=1, dtype=np.int32))
    np.testing.assert_array_equal(scale.view(np.uint32), (maximum.astype(np.float32) * np.float32(8 / 127)).view(np.uint32))
    # Half an output step from rounding the byte, plus at most 8 * 0.54 / 256
    # steps from rounding the factor, including the fast quotient error.
    reconstructed = weights * scale.astype(np.float64)[:, None]
    original = codes * s.T.repeat(G, axis=1)
    assert np.all(np.abs(reconstructed - original) <= 0.52 * scale[:, None])


def test_requant_source():
    src = _source(_requant())
    assert "tl_hvx_load_widened_nibbles(" in src and "tl_hvx_load_repeat<4, 128" in src
    assert "tl_hvx_div(" in src and "roundf(" not in src
    assert "tl_panel_sum<true>" in src and "tl_load_nibble(" not in src
    assert "tl_hvx_high_bytes<int8_t>" in src and "*(int8_t128*)" in src


@pytest.mark.parametrize("dtype", ["uint4", "int4"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("per_row", [False, True])
def test_panel_map_source(dtype, paired, per_row):
    src = _source(_map(dtype, paired, per_row))
    assert "tl_hvx_load_widened_nibbles(" in src and "tl_hvx_load_repeat<4, 128" in src
    assert "*(int8_t128*)" in src and "tl_load_nibble(" not in src


def test_group_of_24_repeats_word_factors_across_four_registers():
    src = _source(_float_factors())
    assert "tl_hvx_load_repeat<4, 128" in src and "tl_hvx_load_widened_nibbles(" in src
    assert "*(int8_t128*)" in src and "tl_load_nibble(" not in src


@pytest.mark.parametrize("paired", [False, True])
def test_byte_panel_to_float_source(paired):
    src = _source(_as_float(paired))
    assert "tl_hvx_load_groups<8," in src and "*(float32*)" in src
    assert "for (int32_t vec" not in src


def test_plain_byte_buffer_keeps_cast_staging():
    from tilelang import tvm
    from tilelang.backend.target import determine_target

    @T.prim_func
    def kernel(A: T.Tensor((32, 4), "int8"), B: T.Tensor((32,), "float32")):
        for i in T.vectorized(32):
            B[i] = T.Cast("float32", A[i, 0])

    target = determine_target("hexagon", return_object=True)
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    lowered = tilelang.transform.DecoupleTypeCast()(mod)
    assert 'T.sblock("decoupled_cast")' in lowered.script()


@pytest.mark.parametrize("paired", [False, True])
def test_panel_sum_source(paired):
    assert f"tl_panel_sum<{str(paired).lower()}>" in _source(_sum(paired, clear=False))
    with pytest.raises(Exception, match="summed along K"):
        _source(_sum(paired, reduce_max=True))


@DEVICE
@pytest.mark.parametrize("paired", [False, True])
def test_panel_sum_device(paired):
    kernel = tilelang.compile(_sum(paired, clear=False), target="hexagon", out_idx=[2])
    rng = torch.Generator().manual_seed(42)
    weights = torch.randint(-128, 128, (64, 128), dtype=torch.int8, generator=rng)
    initial = torch.arange(64, dtype=torch.int32)
    try:
        got = kernel(pack_panel(weights, 8, paired=paired), initial)
        assert torch.equal(got, weights.sum(dim=1, dtype=torch.int32) + initial)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("fast_math", [False, True])
def test_requant_and_hmx_device(fast_math):
    codes, scales = _inputs()
    kernel = tilelang.compile(_requant(), target="hexagon", out_idx=[2, 3, 4, 5], pass_configs={"tl.enable_fast_math": fast_math})
    try:
        packed, scale, sums, factors = kernel(pack_panel(torch.from_numpy(codes), 4), torch.from_numpy(scales))
    finally:
        kernel.adapter.close()
    w = unpack_panel(packed, (N, K), 8, signed=True, paired=True)
    _check_requant(codes, scales, w.numpy(), scale.numpy(), sums.numpy(), factors.numpy())
    gemm = tilelang.compile(_gemm(64, N, K, paired=True), target="hexagon", out_idx=[2])
    try:
        a = torch.randint(0, 256, (1, 64, K), dtype=torch.uint8, generator=torch.Generator().manual_seed(4))
        raw = gemm(a, packed.view(1, N, K))
        assert torch.equal(raw[0].long() - 128 * sums.long(), (a[0].long() - 128) @ w.long().T)
    finally:
        gemm.adapter.close()


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize("fast_math", [False, True])
def test_requant_generated_kernel_on_simulator(arch, fast_math, tmp_path):
    from test_tilelang_hexagon_crouton_regions import _source as arch_source

    # Run the emitted kernel. Standalone simulator programs have no QuRT
    # allocator or DMA service, so model the arena and synchronous transfers.
    header = tmp_path / "tl_templates" / "hexagon" / "dma.h"
    header.parent.mkdir(parents=True)
    (header.parent / "vtcm.h").write_text("#pragma once\n")
    header.write_text("""
#pragma once
#include <string.h>
template<int D, int G> struct tl_hexagon_dma_async_context { int status = 0; };
template<class Q> int tl_hexagon_dma_async_submit(Q &, void *dst, const void *src,
    unsigned ds, unsigned ss, unsigned bytes, unsigned rows, int) {
  for (unsigned r = 0; r < rows; ++r)
    memcpy((char *)dst + r * ds, (const char *)src + r * ss, bytes);
  return 0;
}
template<class Q> int tl_hexagon_dma_async_commit(Q &) { return 0; }
template<class Q> int tl_hexagon_dma_async_wait(Q &, int) { return 0; }
""")
    import tvm

    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})
    source = _source(_requant(), fast_math=fast_math, target=target)
    row_map = arch_source(_as_float(True, N, K), arch)
    cpp = tmp_path / "panel.cc"
    cpp.write_text(
        """
#include <stdio.h>
#include <tl_templates/hexagon/common.h>
alignas(2048) static uint8_t arena[2 << 20];
static uint8_t *tl_vtcm_arena() { return arena; }
static unsigned tl_vtcm_arena_bytes() { return sizeof(arena); }
static unsigned tl_vtcm_shared_high_water;
"""
        + source
        + row_map
        + f"""
alignas(128) static int8_t q[{N * K // 2}], w[{N * K}];
alignas(128) static half s[{K // G * N}];
alignas(128) static float sw[{N}], logical[{N * K}];
alignas(128) static int32_t cs[{N}];
alignas(128) static int16_t factors[{N * K // G}];
int main(int argc, char **argv) {{
  if (argc != 3) return 1;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out || fread(q, sizeof q, 1, in) != 1 || fread(s, sizeof s, 1, in) != 1) return 2;
  if (requant_kernel(q, s, w, sw, cs, factors) || kernel_kernel(w, logical)) return 3;
  if (fwrite(w, sizeof w, 1, out) != 1 || fwrite(sw, sizeof sw, 1, out) != 1 ||
      fwrite(cs, sizeof cs, 1, out) != 1 || fwrite(factors, sizeof factors, 1, out) != 1 || fwrite(logical, sizeof logical, 1, out) != 1) return 4;
  return fclose(out);
}}
"""
    )
    exe = tmp_path / "panel"
    compile_cmd = [
        _tool("hexagon-clang++"),
        f"-m{arch}",
        "-mhvx",
        "-mhvx-length=128B",
        "-O2",
        "-std=c++17",
        "-mllvm",
        "--disable-handle-qfp",
        "-mllvm",
        "-hexagon-insert=false",
        "-I",
        str(tmp_path),
        "-I",
        str(Path(__file__).resolve().parents[3] / "src"),
        str(cpp),
        "-o",
        str(exe),
    ]
    subprocess.run(compile_cmd, check=True, timeout=180)
    codes, scales = _inputs()
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.write_bytes(pack_panel(torch.from_numpy(codes), 4).numpy().tobytes() + scales.tobytes())
    subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(exe), "--", str(inputs), str(outputs)],
        cwd=tmp_path,
        check=True,
        timeout=180,
    )
    data = outputs.read_bytes()
    packed = np.frombuffer(data[: N * K], dtype=np.uint8)
    w = unpack_panel(torch.from_numpy(packed.copy()).view(torch.int8), (N, K), 8, signed=True, paired=True).numpy()
    scale = np.frombuffer(data, "<f4", N, N * K)
    sums = np.frombuffer(data, "<i4", N, N * K + 4 * N)
    factors = np.frombuffer(data, "<i2", N * K // G, N * K + 8 * N).reshape(K // G, N)
    _check_requant(codes, scales, w, scale, sums, factors)
    logical = np.frombuffer(data, "<f4", offset=N * K + 8 * N + factors.nbytes).reshape(N, K)
    np.testing.assert_array_equal(logical, w)
