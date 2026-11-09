"""Host synchronization checks for the actual grid barrier, without a DSP."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def grid_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    directory = tmp_path_factory.mktemp("grid-host")
    headers = directory / "tl_templates" / "hexagon"
    headers.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    worker = (root / "src/tl_templates/hexagon/worker.h").read_text()
    # Retain the barrier/pool implementation. Only the target pause instruction
    # is replaced; the harness supplies the external QuRT/resource interfaces.
    pause = 'asm volatile("pause(#8)");'
    assert pause in worker
    (headers / "worker.h").write_text(worker.replace(pause, "test_pause();"))
    for stub in (directory / "qurt.h", headers / "common.h", headers / "vtcm.h"):
        stub.write_text("")
    binary = directory / "grid-host"
    source = Path(__file__).with_name("grid_sync_host.cc")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-I", str(directory), str(source), "-o", str(binary)],
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "gate-unknown",
        "gate-four",
        "gate-six",
        "visible-four",
        "visible-six-spin",
        "visible-six-blocking",
        "spurious-returns",
        "failed-spin",
        "failed-before-park",
        "failed-parked",
        "completed-before-park",
        "completed-parked",
        "completed-spin",
        "completed-blocking",
    ],
)
def test_grid_barrier_on_host(grid_host, case):
    subprocess.run([str(grid_host), case], check=True, timeout=10)
