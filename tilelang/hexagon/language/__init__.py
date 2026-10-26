"""Hexagon language dialect: common TileLang plus Hexagon extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.allocate import (  # noqa: F401
    alloc_hmx_accumulator,
    alloc_hmx_bias_state,
    alloc_hmx_convert_state,
    alloc_hmx_state,
)
from tilelang.language.builtin import (  # noqa: F401
    hexagon_hmx_acquire,
    hexagon_hmx_clear,
    hexagon_hmx_convert,
    hexagon_hmx_load_bias,
    hexagon_hmx_mma,
    hexagon_hmx_release,
    hexagon_hmx_store,
)
from tilelang.language.copy_op import dma_copy, dma_wait  # noqa: F401

# The Hexagon dialect shadows T.Kernel with a version that accepts
# num_workers (and keeps threads= for serialized tile programs).
from .kernel import Kernel  # noqa: F401

_HEXAGON_API_ALL = (
    "Kernel",
    "alloc_hmx_accumulator",
    "alloc_hmx_bias_state",
    "alloc_hmx_convert_state",
    "alloc_hmx_state",
    "dma_copy",
    "dma_wait",
    "hexagon_hmx_acquire",
    "hexagon_hmx_clear",
    "hexagon_hmx_convert",
    "hexagon_hmx_load_bias",
    "hexagon_hmx_mma",
    "hexagon_hmx_release",
    "hexagon_hmx_store",
)

__tilelang_dialect__ = "hexagon"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_HEXAGON_API_ALL)))

del _COMMON_ALL, _HEXAGON_API_ALL
