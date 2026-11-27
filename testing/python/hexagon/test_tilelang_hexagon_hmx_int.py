"""Exact integer GEMM on HMX and its 64 x 32 activation tiles."""

import json
import logging

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import CroutonLayout, make_hmx_activation_layout, make_hmx_output_layout
from tilelang.hexagon.intrinsics.panel import PanelLayout, make_hvx_panel_layout, pack_panel, unpack_panel
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


def test_paired_panel_bytes_and_positions():
    rows, cols = 128, 96
    shape = (2, rows, cols)
    layout = make_hvx_panel_layout(shape, "int8", paired=True)
    loaded = tvm.ir.load_json(tvm.ir.save_json(layout))
    values = torch.randint(-128, 128, shape, dtype=torch.int8, generator=torch.Generator().manual_seed(1))
    packed = pack_panel(values, 8, paired=True)
    batch, r, c = torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij")
    offsets = (
        batch * rows * cols
        + (r // 64) * 64 * cols
        + (c // 32) * 2048
        + ((r % 64) // 32) * 1024
        + ((c % 32) // 4) * 128
        + (r % 32) * 4
        + c % 4
    )
    assert torch.equal(packed[offsets], values.to(torch.uint8))
    assert torch.equal(unpack_panel(packed, shape, 8, signed=True, paired=True), values.long())
    for lyt in (layout, loaded):
        assert lyt.paired
        for index in [(0, 0, 0), (0, 31, 31), (0, 32, 0), (0, 1, 32), (0, 64, 65), (1, 127, 95)]:
            physical = [int(x) for x in lyt.map_forward_index(index)]
            assert sum(x * stride for x, stride in zip(physical, [64 * cols, 2048, 1024, 128, 4, 1], strict=True)) == offsets[index]


@pytest.mark.parametrize("old_index", [False, True])
def test_legacy_panel_json_defaults_to_unpaired(old_index):
    layout = make_hvx_panel_layout((128, 96), "int8")
    graph = json.loads(tvm.ir.save_json(layout))
    fields = graph["nodes"][graph["root_index"]]["data"]
    fields.pop("paired", None)
    if old_index:
        fields.pop("placeholder_vars")
    loaded = tvm.ir.load_json(json.dumps(graph))
    assert not loaded.paired
    assert [int(x) for x in loaded.map_forward_index([65, 35])] == [2, 8, 1, 3]


@pytest.mark.parametrize("shape,bits", [((96, 64), 8), ((64, 36), 8), ((64, 64), 4)])
def test_invalid_paired_panel(shape, bits):
    with pytest.raises(Exception, match="paired panels require"):
        make_hvx_panel_layout(shape, f"int{bits}", paired=True)
    with pytest.raises(ValueError, match="paired panels require"):
        pack_panel(torch.zeros(shape), bits, paired=True)
    with pytest.raises(ValueError, match="paired panels require"):
        unpack_panel(torch.zeros(shape), shape, bits, signed=True, paired=True)


def test_paired_panel_rejects_vector_elements():
    with pytest.raises(ValueError, match="scalar 8-bit integers"):
        make_hvx_panel_layout((64, 32), "int8x4", paired=True)

    @T.prim_func
    def copy(B: T.Tensor((64, 32), "int8x4"), Out: T.Tensor((64, 32), "int8x4")):
        with T.Kernel(1):
            tmp = T.alloc_shared((64, 32), "int8x4")
            T.annotate_layout({B: PanelLayout(B.shape, 4, paired=True), Out: PanelLayout(Out.shape, 4, paired=True)})
            T.dma_copy(B, tmp)
            T.dma_wait(0)
            T.copy(tmp, Out)

    with pytest.raises(Exception, match="panel copy requires scalar elements"):
        _lower(copy)


def _gemm(M, N, K, *, workers=1, mode="clear", dtype="uint8", row_major=False, offset=0, runtime_rows=False, paired=False):
    batch = 1 if workers == 1 else workers * 2

    @T.prim_func
    def gemm(A: T.Tensor((batch, M + offset, K), dtype), B: T.Tensor((batch, N, K), "int8"), C: T.Tensor((batch, M, N), "int32")):
        with T.Kernel(batch, num_workers=workers) as w:
            a = T.alloc_shared((M + offset, K), dtype)
            b = T.alloc_shared((N, K), "int8")
            c = T.alloc_fragment((M, N), "int32")
            T.annotate_layout({B: make_hvx_panel_layout(B, paired=paired)})
            if row_major:
                T.annotate_layout({a: T.Layout(a.shape, lambda i, j: [i, j])})
            T.copy(A[w, :, :], a)
            if paired and mode == "clear":
                T.dma_copy(B[w, :, :], b)
                T.dma_wait(0)
            else:
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


PAIRED_CASES = [
    ((64, 64, 32), "clear", 1),
    ((128, 256, 2048), "clear", 1),
    ((64, 128, 8192), "clear", 1),
    ((64, 128, 96), "add", 1),
    ((128, 128, 128), "split", 2),
]
ALL_CASES = [(*case, False) for case in CASES] + [(*case, True) for case in PAIRED_CASES]


@pytest.mark.parametrize("shape,mode,workers,paired", ALL_CASES)
def test_integer_hmx_lowering(shape, mode, workers, paired):
    src = _lower(_gemm(*shape, mode=mode, workers=workers, paired=paired)).kernel_source
    assert f"tl_hexagon_hmx_gemm_int<{'true' if workers > 1 else 'false'}, {str(paired).lower()}>(" in src
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


def _local_weights(n, start=0, explicit=None, backing_k=64, scope="shared.dyn"):
    @T.prim_func
    def gemm(A: T.Tensor((64, 64), "uint8"), C: T.Tensor((64, n), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "uint8")
            b = T.sblock_alloc_buffer((128, backing_k), "int8", scope=scope)
            c = T.alloc_fragment((64, n), "int32")
            if explicit is not None:
                T.annotate_layout({b: make_hvx_panel_layout(b, paired=explicit)})
            T.clear(b)
            T.copy(A, a)
            T.gemm(a, b[start : start + n, :64], c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return gemm


@pytest.mark.parametrize(
    "n,start,explicit,backing_k,paired",
    [
        (64, 64, None, 64, True),
        (64, 0, False, 64, False),
        (64, 0, True, 64, True),
        (96, 0, None, 64, False),
        (64, 32, None, 64, False),
        (64, 0, None, 68, False),
    ],
)
def test_local_weight_layout_selection(n, start, explicit, backing_k, paired):
    source = _lower(_local_weights(n, start, explicit, backing_k)).kernel_source
    assert f"tl_hexagon_hmx_gemm_int<false, {str(paired).lower()}>(" in source


def test_grid_vtcm_respects_ordinary_weights():
    source = _lower(_local_weights(64, explicit=False, scope="global.vtcm")).kernel_source
    assert "tl_hexagon_hmx_gemm_int<false, false>(" in source


@pytest.mark.parametrize("n,start", [(32, 0), (64, 32)])
def test_explicit_pairs_require_whole_pair_region(n, start):
    with pytest.raises(Exception, match="whole 64-row pairs"):
        _lower(_local_weights(n, start, explicit=True))


def test_hvx_rejects_paired_weights():
    with pytest.raises(Exception, match="HVX reads .* in unpaired panels"):
        _lower(_gemm(32, 64, 64, paired=True))


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
    line = next(line for line in source.splitlines() if "tl_hexagon_hmx_gemm_int<true, false>" in line)
    assert line.endswith(", true);")


def test_integer_hmx_rejects_runtime_rows():
    with pytest.raises(Exception, match="does not support hexagon.rows"):
        _lower(_gemm(64, 32, 32, runtime_rows=True))


@DEVICE
@pytest.mark.parametrize("shape,mode,workers,paired", ALL_CASES)
def test_exact_integer_hmx(shape, mode, workers, paired):
    M, N, K = shape
    batch = 1 if workers == 1 else workers * 2
    g = torch.Generator().manual_seed(K)
    a = torch.randint(0, 256, (batch, M, K), dtype=torch.uint8, generator=g)
    b = torch.randint(-128, 128, (batch, N, K), dtype=torch.int8, generator=g)
    # Exercise carries into all four bytes and both signs at large K.
    a[:, 0] = 255
    b[:, 0], b[:, 1] = -128, 127
    if paired:
        b[:, 32], b[:, 33] = 127, -128
    kernel = tilelang.compile(_gemm(*shape, mode=mode, workers=workers, paired=paired), out_idx=[2], target="hexagon")
    try:
        actual = kernel(a, pack_panel(b, 8, paired=paired))
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


def _paired_dma(row=64, col=32, dst_paired=True):
    @T.prim_func
    def copy(B: T.Tensor((2, 192, 128), "int8"), Out: T.Tensor((128, 64), "int8")):
        with T.Kernel(1):
            tmp = T.alloc_shared((128, 64), "int8")
            T.annotate_layout(
                {
                    B: make_hvx_panel_layout(B, paired=True),
                    tmp: make_hvx_panel_layout(tmp, paired=dst_paired),
                    Out: make_hvx_panel_layout(Out, paired=dst_paired),
                }
            )
            T.dma_copy(B[1, row : row + 128, col : col + 64], tmp)
            T.dma_wait(0)
            T.copy(tmp, Out)

    return copy


def test_strided_paired_dma():
    source = _lower(_paired_dma()).kernel_source
    assert "(uint32_t)4096, (uint32_t)8192, (uint32_t)4096, (uint32_t)2" in source


@pytest.mark.parametrize(
    "row,col,dst_paired,reason",
    [(32, 32, True, "multiples of 64"), (64, 4, True, "multiples of 32"), (64, 32, False, "matching paired or unpaired")],
)
def test_paired_dma_checks_both_regions(row, col, dst_paired, reason):
    with pytest.raises(Exception, match=reason):
        _lower(_paired_dma(row, col, dst_paired))


@DEVICE
def test_strided_paired_dma_bytes():
    b = torch.randint(-128, 128, (2, 192, 128), dtype=torch.int8)
    kernel = tilelang.compile(_paired_dma(), out_idx=[1], target="hexagon")
    try:
        actual = kernel(pack_panel(b, 8, paired=True))
    finally:
        kernel.adapter.close()
    assert torch.equal(actual.flatten().view(torch.uint8), pack_panel(b[1, 64:, 32:96], 8, paired=True))
