"""Crouton layouts: the native VTCM layouts of HMX operands.

A Crouton tile is a 32x32 FP16 block stored as interleaved row pairs: element
``(row, col)`` sits at ``(row // 2) * 64 + col * 2 + row % 2``, so one HVX
register holds one row pair. HMX operands tile a matrix with these blocks in an
operand-specific order, flat in memory:

* activation ``[M, K]`` and output ``[M, N]``: tiles by rows, ``[mt, kt]``;
* weight ``[K, N]``: tiles by columns, ``[nt, kt]`` (K tiles contiguous).

An 8-bit activation or output tile has 64 rows and 32 columns, with four
rows interleaved per register. Leading dimensions stack whole matrices. A ``CroutonLayout`` is a ``T.Layout``
that lowerings recognize by its type (``src/hexagon/layout/crouton.h``).
"""

from __future__ import annotations

from collections.abc import Sequence

import tvm_ffi
from tvm import DataType

from tilelang.layout import Layout

TILE = 32
TILE_ELEMS = TILE * TILE

__all__ = [
    "TILE",
    "TILE_ELEMS",
    "CroutonLayout",
    "make_hmx_crouton_layout",
    "make_hmx_activation_layout",
    "make_hmx_weight_layout",
    "make_hmx_output_layout",
]


@tvm_ffi.register_object("tl.hexagon.CroutonLayout")
class CroutonLayout(Layout):
    """Crouton tiles over the last two dimensions of a buffer.

    Parameters
    ----------
    shape : the buffer shape; the matrix is its last two dimensions, swapped
        when ``transposed``.
    column_tiles : order tiles by the matrix's columns (HMX weights) rather
        than its rows (activations and outputs).
    transposed : the buffer stores the transposed matrix.
    bits : element width, 8 for byte activations and outputs or 16 for fp16.
    """

    def __init__(self, shape, column_tiles: bool, transposed: bool = False, bits: int = 16):
        self.__init_handle_by_constructor__(
            tvm_ffi.get_global_func("tl.hexagon.CroutonLayout"), list(shape), column_tiles, transposed, bits
        )


def _shape_of(buffer_or_shape) -> tuple:
    shape = getattr(buffer_or_shape, "shape", buffer_or_shape)
    if not isinstance(shape, Sequence):
        raise TypeError("HMX layout expects a Buffer or a shape sequence")
    return tuple(shape)


def _static_extent(value, name: str) -> int:
    value = getattr(value, "value", value)
    if not isinstance(value, int):
        raise ValueError(f"HMX {name} extent must be static; got {value}")
    return value


def _bits(buffer_or_shape) -> int:
    return 8 if DataType(str(getattr(buffer_or_shape, "dtype", "float16"))).bits == 8 else 16


def _validate(shape: tuple, name: str, transposed: bool, bits: int) -> None:
    if bits == 8 and (transposed or name == "weight"):
        raise ValueError("8-bit Crouton requires row-ordered, non-transposed tiles")
    if len(shape) < 2:
        raise ValueError(f"HMX {name} layout requires at least two dimensions")
    rows = _static_extent(shape[-1 if transposed else -2], f"{name} rows")
    cols = _static_extent(shape[-2 if transposed else -1], f"{name} columns")
    tile_rows = 64 if bits == 8 else TILE
    if rows <= 0 or cols <= 0 or rows % tile_rows or cols % TILE:
        raise ValueError(f"HMX {name} dimensions must be positive multiples of {tile_rows} rows and {TILE} columns; got ({rows}, {cols})")
    for i, extent in enumerate(shape[:-2]):
        _static_extent(extent, f"{name} prefix dimension {i}")


def make_hmx_crouton_layout() -> CroutonLayout:
    """Return the native 32x32 FP16 HMX Crouton layout atom."""

    return CroutonLayout((TILE, TILE), column_tiles=False)


def make_hmx_activation_layout(buffer_or_shape, transposed: bool = False) -> CroutonLayout:
    """The activation layout: tiles ``[..., mt, kt]``.

    The normal logical shape is ``[..., M, K]``. With ``transposed=True`` the
    buffer is ``[..., K, M]``, but its tiles keep the HMX activation order of
    the mathematical ``[M, K]`` operand.
    """

    shape = _shape_of(buffer_or_shape)
    bits = _bits(buffer_or_shape)
    _validate(shape, "activation", transposed, bits)
    return CroutonLayout(shape, column_tiles=False, transposed=transposed, bits=bits)


def make_hmx_weight_layout(buffer_or_shape, transposed: bool = False) -> CroutonLayout:
    """The weight layout: tiles ``[..., nt, kt]``.

    The normal logical shape is ``[..., K, N]``. With ``transposed=True`` the
    buffer is ``[..., N, K]``, while HMX still sees mathematical ``[K, N]``.
    """

    shape = _shape_of(buffer_or_shape)
    bits = _bits(buffer_or_shape)
    _validate(shape, "weight", transposed, bits)
    return CroutonLayout(shape, column_tiles=True, transposed=transposed)


def make_hmx_output_layout(buffer_or_shape) -> CroutonLayout:
    """The output layout of logical ``[..., M, N]``: tiles ``[..., mt, nt]``."""

    shape = _shape_of(buffer_or_shape)
    bits = _bits(buffer_or_shape)
    _validate(shape, "output", False, bits)
    return CroutonLayout(shape, column_tiles=False, bits=bits)
