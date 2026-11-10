"""Hexagon dialect of ``T.Kernel``: the common launch plus Hexagon launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
    num_workers: int | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Hexagon: a grid of tile programs.

    The grid becomes a loop nest over the tile programs, run by one DSP
    hardware thread, or by a worker pool when ``num_workers`` is given. There
    are no SIMT threads, so this dialect has no ``threads`` and
    ``T.get_thread_binding()`` is rejected at compile time.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        program index per axis.
    prelude : str, optional
        C source injected before the generated kernel, e.g. ``#include``
        lines or helper functions.
    num_workers : int, optional
        Requested upper bound on workers, capped by the device's reported HW
        threads, 128-byte HVX contexts, and the SDK's ``QURT_MAX_HTHREAD_LIMIT``
        storage capacity (currently 8). Unavailable queries fall back to the
        calling thread alone. Each worker runs whole
        programs with its own buffers and VTCM arena. Programs must be independent
        except at ``T.sync_grid()``: its grid must fit both ``num_workers`` and
        the available workers and arenas, or fail with ``TL_ERR_GRID`` before
        any program starts. ``None`` runs the grid on one thread.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(M, 32), num_workers=6) as bx:
            ...

        # Requires capacity for two simultaneous workers.
        # Every program writes its slice of `h`, then reads all of it.
        with T.Kernel(2, num_workers=2) as w:
            ...
            T.sync_grid()
            ...
    """
    annotations = {}
    # Validate here so a typo raises a clean ValueError instead of a later
    # compiler error (bool is an int subclass, so reject it explicitly).
    if num_workers is not None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"T.Kernel(num_workers=...) must be a positive int, got {num_workers!r}")
        annotations["hexagon.num_workers"] = num_workers
    return launch_kernel(blocks, prelude=prelude, **annotations)
