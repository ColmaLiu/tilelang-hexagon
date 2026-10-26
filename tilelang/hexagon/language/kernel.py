"""Hexagon dialect of ``T.Kernel``: the common launch plus Hexagon launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
    num_workers: int | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Hexagon: a grid of tile programs.

    The Hexagon codegen serializes the grid (and any ``threads``) into nested
    loops on one DSP hardware thread, or fans the outermost grid axis across a
    worker pool when ``num_workers`` is given.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        program index per axis.
    threads : int | list[int] | tuple[int, ...], optional
        Serialized thread extents of each tile program. Defaults to 1.
    prelude : str, optional
        C source injected before the generated kernel, e.g. ``#include``
        lines or helper functions.
    num_workers : int, optional
        Distribute the outermost grid axis across this many HW threads (the
        worker pool, one VTCM slice per worker). ``None`` keeps the grid
        serial.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(M, 32), threads=1, num_workers=6) as bx:
            ...
    """
    annotations = {}
    # Propagated to the device func like cluster_dims and read by
    # CodeGenTileLangHexagon. Validate here so a typo raises a clean ValueError
    # instead of a downstream LOG(FATAL) / silent serial fallback (bool is
    # rejected explicitly — it is an int subclass).
    if num_workers is not None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"T.Kernel(num_workers=...) must be a positive int, got {num_workers!r}")
        annotations["hexagon.num_workers"] = num_workers
    return launch_kernel(blocks, threads=threads, prelude=prelude, **annotations)
