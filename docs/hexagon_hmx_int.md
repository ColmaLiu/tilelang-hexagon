# Integer HMX

This note records what the HMX of v73 and v79 offers beyond fp16, what was
measured on the devices, and how TileLang maps it: `T.gemm` of a `uint8` A by
an `int8` B into an exact `int32` C (#30).

## What the SDK exposes

`hmx_hexagon_protos.h` (Hexagon Tools 19.0.07) lists these operand types:

| port | types |
|---|---|
| activation | `hf` (fp16), `ub` (unsigned 8-bit), `f8` |
| weight | `hf`, `b` (signed 8-bit), `n` (4-bit, also `:2x`), `c`, `sc`, `sm`, `sbit`, `ubit`, `f8` |
| conversion | `cvt.hf`, `cvt.ub` (one byte of the accumulator), `cvt.uh` (`:2x1`, `:2x2`), `cvt.f8` |

The activation port is unsigned for integers: a signed operand goes through a
zero point. fp8 is not pursued here (it compiles for v79 but not for v73).
4-bit weights (`weight.n`) multiply 4-bit codes directly, but one accumulation
has one scale per output column, so they suit per-channel 4-bit weights, not
the per-group scales of the LFM2 bundle.

## Measured rates

A probe of the bare instructions: single thread, operands in VTCM, a
64-tile-deep (K = 2048) accumulation per output block, cycles per 32 x 32 x 32
multiply-accumulate. "With output" includes the conversions and stores of each
block (one for fp16, two bytes for int8).

| | v79 (phone) | v73 (SA8650) |
|---|---:|---:|
| fp16, MACs | 7.79 | 15.60 |
| fp16, with output | 8.00 | 15.87 |
| int8, 32 columns (`weight.b`), MACs | 3.34 | 2.90 |
| int8, 32 columns, with output | 3.42 | 3.16 |
| int8, 64 columns (`weight.b:deep`), MACs | 2.07 | 1.96 |
| int8, 64 columns, with output | 2.17 | 2.13 |

Int8 runs 2.3x (32 columns) to 3.7x (64 columns) as fast as fp16 on v79, and
5x to 7.5x on v73, whose fp16 rate is half v79's.

## Layouts and conversion (verified bit-exact on both devices)

- **Activation tile**: 64 rows x 32 K of `uint8`, 2048 bytes; element
  `(r, k)` at `(r // 4) * 128 + k * 4 + r % 4`: one HVX register holds four
  rows. A deep load reads consecutive tiles along K (range `2048 n - 1`).
- **Weight tile** (`weight.b`): 32 K x 32 columns of `int8`, 1024 bytes;
  element `(k, n)` at `(k // 4) * 128 + n * 4 + k % 4`. This is the 8-bit
  **panel layout** that HVX's `vrmpy` already reads (`tilelang/hexagon/intrinsics/panel.py`):
  a panel's 32 rows of B are consecutive weight tiles along K.
- **64 columns** (`weight.b:deep`): each K step reads 2048 bytes, the tiles of
  two column groups one after the other. Their products go to two
  accumulators, converted in turn.
- **Output**: `cvt.ub` takes one byte of each accumulator, after adding a
  per-column bias and shifting by a per-column amount, both from a bias bank
  (`bias = mxmem2`, 256 bytes: 32 exponent words, then 32 int32 biases). v79
  holds four banks and selects one in the conversion's control word; v73 has
  one, reloaded before each conversion. Exponent `e` selects bits
  `[24 - e, 32 - e)`: 24, 16, 8 and 0 give the four bytes. With control bit 1
  set, the byte wraps instead of saturating, and the four bytes are the exact
  two's-complement int32 (checked at K = 2048 with full-range operands). The
  output tile is stored spatially, in the activation tile layout, with store
  range `0xF00` (the fp16 range `0x700` writes only its first 32 rows).
- llama.cpp (ggml-hexagon) keeps 16 bits instead: two bytes of
  `(acc + bias) >> p`, with `p` from each column's bound and the bias folding
  in its activation zero point. That is also bit-exact as specified, but drops
  `p` low bits.

## `T.gemm` on integer HMX

`T.gemm(A, B, C, transpose_B=True)` runs on HMX when:

- A is `uint8` `[M, K]` in VTCM, in the 8-bit activation layout (an 8-bit
  `CroutonLayout`: 64 x 32 tiles, four rows a register);
- B is `int8` `[N, K]` in VTCM, in the 8-bit panel layout;
- C is `int32`, row-major, in VTCM;
- M is a multiple of 64, and N and K of 32, and each region starts on a tile.

C is exact: `C (+)= A @ B^T` in int32 (each call wraps like int32 arithmetic,
which a K below 2^31 / (255 * 128) = 65793 never reaches). It is the same
result as the HVX path (`tl_hvx_gemm_int`); other integer GEMMs stay there,
under its own rules (whole panels of B).

A call (`tl_hexagon_hmx_gemm_int`, `hmx_gemm_int.h`) multiplies each 64 x 32
output tile over its whole K in the accumulator, converts its four bytes into
staging planes, and an HVX epilogue joins them into rows of C while HMX
multiplies the next tile. In a worker pool each tile is a critical section, as
for fp16.

`T.gemm` on the devices, operands in VTCM, cycles per 32 x 32 x 32 over 50
calls (fp16 into an fp32 C, int8 into an int32 C):

| M x N x K | v79 fp16 | v79 int8 | v73 fp16 | v73 int8 |
|---|---:|---:|---:|---:|
| 128 x 256 x 2048 | 8.35 | 3.76 | 16.34 | 3.77 |
| 256 x 256 x 2048 | 8.29 | 3.65 | 16.30 | 3.65 |
| 128 x 256 x 8192 | 8.12 | 3.56 | 16.08 | 3.20 |

The 64-column weight form (`weight.b:deep`) would take the int8 rate toward
2.2; it needs B's panels paired per K tile, a layout of its own.

Why A is unsigned: HMX has no signed activation. A signed A would need its
buffer to hold `a ^ 0x80` instead of `a`, a value encoding that TileLang's
layouts (index maps) do not express. Quantized activations usually carry a
zero point anyway. For `A = q + 128`, `A @ B^T = q @ B^T + 128 * colsum(B)`;
the caller subtracts the column sums, which are constants for weights.

## Using it for a quantized prefill

LFM2's prefill GEMMs keep fp16 activations today and dequantize the 4-bit
weights to fp16 tiles on HVX (`tl_crouton_dequant4`). An integer prefill
quantizes each activation row to uint8 and requantizes each 4-bit weight group
into int8 per output channel (the group scales do not fit one accumulation).
Both are lossy; the model-level study and the measured prefill are in the LFM2
example's documentation.
