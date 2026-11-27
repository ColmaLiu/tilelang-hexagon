"""Exact integer GEMM on HMX and its 64 x 32 activation tiles."""

import json
import logging

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import CroutonLayout, make_hmx_activation_layout, make_hmx_output_layout
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel
from test_tilelang_hexagon_gemm_int import DEVICE, _lower


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("legacy", [False, True])
def test_crouton_registers(bits, legacy):
    layout = CroutonLayout((2, 128, 96), False, bits=bits)
    graph = json.loads(tvm.ir.save_json(layout))
    if legacy:
        fields = graph["nodes"][graph["root_index"]]["data"]
        fields.pop("placeholder_vars")
        if bits == 16:
            fields.pop("bits", None)
    loaded = tvm.ir.load_json(json.dumps(graph))
    group = 32 // bits
    height = 16 * group
    for lyt in (layout, loaded):
        assert lyt.bits == bits
        for b, r, c in [(0, 0, 0), (1, 63, 31), (0, 64, 32), (1, 127, 95), (0, 13, 55)]:
            tile = (b * (128 // height) + r // height) * 3 + c // 32
            assert [int(x) for x in lyt.map_forward_index([b, r, c])] == [tile, r % height // group, (c % 32) * group + r % group]


@pytest.mark.parametrize("column,transpose", [(True, False), (False, True)])
def test_byte_layout_requires_activation_order(column, transpose):
    with pytest.raises(Exception, match="row-ordered, non-transposed"):
        CroutonLayout((64, 64), column, transpose, bits=8)


def _gemm(M, N, K, *, workers=1, mode="clear", dtype="uint8", row_major=False, offset=0, runtime_rows=False):
    batch = 1 if workers == 1 else workers * 2

    @T.prim_func
    def gemm(A: T.Tensor((batch, M + offset, K), dtype), B: T.Tensor((batch, N, K), "int8"), C: T.Tensor((batch, M, N), "int32")):
        with T.Kernel(batch, num_workers=workers) as w:
            a = T.alloc_shared((M + offset, K), dtype)
            b = T.alloc_shared((N, K), "int8")
            c = T.alloc_fragment((M, N), "int32")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            if row_major:
                T.annotate_layout({a: T.Layout(a.shape, lambda i, j: [i, j])})
            T.copy(A[w, :, :], a)
            T.copy(B[w, :, :], b)
            if mode == "split":
                for k in T.serial(2):
                    T.gemm(
                        a[:, k * (K // 2) : (k + 1) * (K // 2)],
                        b[:, k * (K // 2) : (k + 1) * (K // 2)],
                        c,
                        transpose_B=True,
                        clear_accum=k == 0,
                    )
            else:
                T.gemm(a[offset:, :], b, c, transpose_B=True, clear_accum=True, annotations={"hexagon.rows": M} if runtime_rows else {})
                if mode == "add":
                    T.gemm(a, b, c, transpose_B=True, clear_accum=False)
            T.copy(c, C[w, :, :])

    return gemm


SHAPES = [(64, 32, 32), (128, 256, 2048), (64, 64, 8192), (192, 96, 1056)]
CASES = [(shape, "clear", 1) for shape in SHAPES] + [((128, 96, 128), "add", 1), ((128, 96, 128), "split", 2)]


@pytest.mark.parametrize("shape,mode,workers", CASES)
def test_integer_hmx_lowering(shape, mode, workers):
    src = _lower(_gemm(*shape, mode=mode, workers=workers)).kernel_source
    assert f"tl_hexagon_hmx_gemm_int<{'true' if workers > 1 else 'false'}>(" in src
    assert "#define TL_HEXAGON_USES_HMX 1" in src
    assert "tl_crouton_copy<true, false>" in src
    assert "tl_hvx_gemm_int<" not in src


@pytest.mark.parametrize(
    "shape,kwargs,reason",
    [
        ((32, 32, 64), {}, "multiples of 64"),
        ((64, 32, 36), {}, "N and K in multiples of 32"),
        ((64, 32, 64), {"dtype": "int8"}, "needs uint8 A x int8 B"),
        ((64, 32, 64), {"row_major": True}, "explicit row-major"),
        ((64, 32, 64), {"offset": 32}, "does not start on a 64 x 32 tile"),
    ],
)
def test_integer_hvx_fallback(shape, kwargs, reason, caplog):
    with caplog.at_level(logging.DEBUG, logger="tilelang.hexagon.op.gemm"):
        assert "tl_hvx_gemm_int<" in _lower(_gemm(*shape, **kwargs)).kernel_source
    assert reason in caplog.text


def test_integer_hmx_checks_vtcm_and_output_layout(caplog):
    def make(output):
        @T.prim_func
        def gemm(A: T.Tensor((64, 32), "uint8"), B: T.Tensor((32, 32), "int8"), C: T.Tensor((64, 32), "int32")):
            with T.Kernel(1):
                a = T.alloc_shared((64, 32), "uint8")
                b = T.alloc_shared((32, 32), "int8")
                c = T.alloc_shared((64, 32), "int32")
                T.annotate_layout({B: make_hvx_panel_layout(B)})
                T.copy(B, b)
                T.copy(A, a)
                if output == "strided":
                    T.annotate_layout({c: T.Layout(c.shape, lambda i, j: [j, i])})
                if output == "global":
                    T.gemm(a, b, C, transpose_B=True, clear_accum=True)
                else:
                    T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                    T.copy(c, C)

        return gemm

    with caplog.at_level(logging.DEBUG, logger="tilelang.hexagon.op.gemm"):
        assert "tl_hvx_gemm_int<" in _lower(make("global")).kernel_source
    assert "C (C) is not in VTCM" in caplog.text
    with pytest.raises(Exception, match="requires a row-major int32 C"):
        _lower(make("strided"))


def test_integer_worker_reloads_banks_if_another_worker_can_use_fp16():
    @T.prim_func
    def mixed(X: T.Tensor((2, 64, 32), "uint8"), B: T.Tensor((32, 32), "int8"), C: T.Tensor((2, 64, 32), "int32")):
        with T.Kernel(2, num_workers=2) as w:
            a = T.alloc_shared((64, 32), "uint8")
            b = T.alloc_shared((32, 32), "int8")
            c = T.alloc_fragment((64, 32), "int32")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            T.copy(X[w, :, :], a)
            T.copy(B, b)
            if w == 1:
                fa = T.alloc_shared((32, 32), "float16")
                fb = T.alloc_shared((32, 32), "float16")
                fc = T.alloc_shared((32, 32), "float16")
                T.clear(fa)
                T.clear(fb)
                T.gemm(fa, fb, fc, clear_accum=True)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C[w, :, :])

    source = _lower(mixed).kernel_source
    assert "tl_hexagon_hmx_gemm<" in source
    line = next(line for line in source.splitlines() if "tl_hexagon_hmx_gemm_int<true>" in line)
    assert line.endswith(", true);")


def test_integer_hmx_rejects_runtime_rows():
    with pytest.raises(Exception, match="does not support hexagon.rows"):
        _lower(_gemm(64, 32, 32, runtime_rows=True))


@DEVICE
@pytest.mark.parametrize("shape,mode,workers", CASES)
def test_exact_integer_hmx(shape, mode, workers):
    M, N, K = shape
    batch = 1 if workers == 1 else workers * 2
    g = torch.Generator().manual_seed(K)
    a = torch.randint(0, 256, (batch, M, K), dtype=torch.uint8, generator=g)
    b = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8, generator=g)
    # Exercise carries into all four bytes and both signs at large K.
    a[:, 0] = 255
    b[:, 0], b[:, 1] = -128, 127
    kernel = tilelang.compile(_gemm(*shape, mode=mode, workers=workers), out_idx=[2], target="hexagon")
    try:
        actual = kernel(a, pack_panel(b, 8))
    finally:
        kernel.adapter.close()
    reference = a.long() @ b.long().transpose(-2, -1)
    assert torch.equal(actual.long(), reference * (2 if mode == "add" else 1))


def _byte_copy(dtype, dma_rows=64):
    shape = (192, 96)

    @T.prim_func
    def copy(X: T.Tensor(shape, dtype), Packed: T.Tensor(shape, dtype), Y: T.Tensor((64, 32), dtype)):
        with T.Kernel(1):
            tiles = T.alloc_shared(shape, dtype)
            tmp = T.alloc_shared((64, 32), dtype)
            T.annotate_layout(
                {tiles: make_hmx_activation_layout(tiles), Packed: make_hmx_output_layout(Packed), tmp: make_hmx_activation_layout(tmp)}
            )
            T.copy(X, tiles)
            T.dma_copy(tiles, Packed)
            T.dma_wait(0)
            T.dma_copy(Packed[dma_rows : dma_rows + 64, 32:64], tmp)
            T.dma_wait(0)
            T.copy(tmp, Y)

    return copy


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_byte_crouton_copy_and_dma_lowering(dtype):
    src = _lower(_byte_copy(dtype)).kernel_source
    assert "tl_crouton_copy<true, false>" in src and "tl_crouton_copy<false, false>" in src
    assert "(uint32_t)18432" in src and "(uint32_t)2048" in src


def test_byte_crouton_dma_requires_64_rows():
    with pytest.raises(Exception, match="multiples of 64"):
        _lower(_byte_copy("uint8", dma_rows=32))


@DEVICE
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_byte_crouton_copy_matches_packed_bytes(dtype):
    x = torch.randint(0, 256, (192, 96), dtype=torch.uint8).to(getattr(torch, dtype))
    kernel = tilelang.compile(_byte_copy(dtype), out_idx=[1, 2], target="hexagon")
    try:
        packed, y = kernel(x)
    finally:
        kernel.adapter.close()
    expected = x.reshape(3, 16, 4, 3, 32).permute(0, 3, 1, 4, 2).contiguous()
    assert torch.equal(packed.flatten(), expected.flatten())
    assert torch.equal(y, x[64:128, 32:64])
