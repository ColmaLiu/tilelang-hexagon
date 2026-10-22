"""Docker SDK preparation and launcher configuration, without Docker or a board."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker/hexagon"


@pytest.mark.parametrize("preinstalled", [False, True])
def test_prepare_sdk_installs_qaic_for_unprivileged_runtime(tmp_path, preinstalled):
    spec = importlib.util.spec_from_file_location("prepare_sdk", DOCKER / "prepare-sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qaic_dir = tmp_path / "ipc/fastrpc/qaic"
    location = qaic_dir / ("bin" if preinstalled else "Ubuntu")
    location.mkdir(parents=True)
    (location / "qaic").write_text("#!/bin/sh\nprintf 'qaic ready\\n'\n")
    (location / "qaic").chmod(0o750)
    if not preinstalled:
        (qaic_dir / "Makefile").write_text("all:\n\tmkdir -p bin\n\tcp -p Ubuntu/qaic bin/qaic\n")
    helpers = tmp_path / "utils/examples"
    helpers.mkdir(parents=True)
    for suffix in ("c", "h"):
        (helpers / f"dsp_capabilities_utils.{suffix}").touch()
    setup = tmp_path / "setup_sdk_env.source"
    setup.write_text("make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/ clean\nmake -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/\n")
    module.prepare_sdk(tmp_path)
    qaic = qaic_dir / "bin/qaic"
    assert qaic.stat().st_mode & 0o777 == 0o755
    assert subprocess.check_output([str(qaic)], text=True) == "qaic ready\n"
    assert "make -C" not in setup.read_text()
    module.prepare_sdk(tmp_path)  # Repeated preparation keeps the installed tool.
    (helpers / "dsp_capabilities_utils.c").unlink()
    with pytest.raises(RuntimeError, match="supply the full SDK"):
        module.prepare_sdk(tmp_path)


@pytest.mark.parametrize(
    "hlos,remote,expected",
    [
        ("linux", None, "/tmp/tilelang_sample_hexagon"),
        ("linux", "/custom/path with spaces", "/custom/path with spaces"),
        ("android", None, "/data/local/tmp/tilelang_sample_hexagon"),
    ],
)
def test_launcher_preserves_remote_override_and_image_name(tmp_path, hlos, remote, expected):
    # Execute the real launcher and capture Docker argv instead of starting it.
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if sys.argv[1:3] == ['container', 'inspect'] else 0)\n"
    )
    fake_docker.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    environ = {k: v for k, v in os.environ.items() if not k.startswith("TILELANG_HEXAGON_") and k != "IMAGE"}
    environ.update(PATH=str(tmp_path) + os.pathsep + environ["PATH"], DOCKER_CALLS=str(calls), TILELANG_HEXAGON_HLOS=hlos)
    if remote:
        environ["TILELANG_HEXAGON_REMOTE_DIR"] = remote
    subprocess.run(["bash", str(DOCKER / "enter-docker.sh"), "sample"], cwd=tmp_path, env=environ, check=True)
    run = next(call for call in map(json.loads, calls.read_text().splitlines()) if call[0] == "run")
    assert f"TILELANG_HEXAGON_REMOTE_DIR={expected}" in run
    assert "tilelang-hexagon:sa8650-v73" in run
    assert "TMPDIR=/tmp" in run


def test_android_connect_board_ignores_stale_ssh_environment(tmp_path):
    environ = dict(os.environ, TILELANG_HEXAGON_HLOS="android", TILELANG_HEXAGON_SSH_HOST="root@unused")
    # No ADB/SSH setup should be attempted, even without a known_hosts variable.
    environ.pop("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", None)
    subprocess.run(["bash", str(DOCKER / "connect-board.sh")], env=environ, cwd=tmp_path, check=True)
