"""Local build identity, offline guards and strict Session shutdown for the bench."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deny(*args, **kwargs):
    raise RuntimeError("Offline preparation/verification cannot open a device or network transport")


def environment(repo, arch, hlos, out, offline):
    assert (arch, hlos) in (("v73", "linux"), ("v79", "android"))
    sys.dont_write_bytecode = True
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="1",
        HEXAGON_DSP_ARCH=arch,
        DSP_ARCH=arch,
        TILELANG_HEXAGON_HLOS=hlos,
        TL_TEMPLATE_PATH=str(out / "template/src"),
    )
    for name in ("cache", "tmp"):
        (out / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TILELANG_CACHE_DIR=str(out / "cache"), XDG_CACHE_HOME=str(out / "cache"), TMPDIR=str(out / "tmp"))
    if offline:
        for name in ("TL_HEXAGON_DEVICE_TEST", "ANDROID_SERIAL", "TILELANG_HEXAGON_DEVICE", "TILELANG_HEXAGON_SSH_HOST"):
            os.environ.pop(name, None)
    # Keep the caller's Linux cross compiler, BSP libraries, legacy-resource and
    # CRC settings. No machine-specific toolchain or device defaults belong here.
    sys.path.insert(0, str(repo))


@contextlib.contextmanager
def offline():
    """Reject Python networking and every runtime device transport used here.

    The SDK compiler/build subprocesses remain available. They use installed
    local tools; this guard is not an OS sandbox for arbitrary build scripts.
    """
    with contextlib.ExitStack() as stack:
        for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
            stack.enter_context(patch.object(socket, name, deny))
        for name in ("connect", "connect_ex", "bind", "listen", "accept", "send", "sendall", "sendto", "sendmsg"):
            if hasattr(socket.socket, name):
                stack.enter_context(patch.object(socket.socket, name, deny))
        from tilelang.hexagon.runtime import session, device, linux_runtime

        stack.enter_context(patch.object(session.Session, "_open", deny))
        stack.enter_context(patch.object(device.AdbConnection, "_run_process", deny))
        stack.enter_context(patch.object(linux_runtime.LinuxConnection, "from_environment", deny))
        # Linux shell/upload/forwarding methods can spawn ssh directly.
        for cls in (device.AdbConnection, linux_runtime.LinuxConnection):
            for name in (
                "list_devices",
                "shell",
                "push",
                "pull",
                "forward_port",
                "remove_forward",
                "start_shell",
                "run_host_binary",
                "start_host_binary",
                "cleanup",
            ):
                stack.enter_context(patch.object(cls, name, deny))
        yield


def build_environment():
    from tilelang.hexagon.runtime.build import _RUNTIME_ONLY

    return {key: value for key, value in sorted(os.environ.items()) if key.startswith("TILELANG_HEXAGON_") and key not in _RUNTIME_ONLY}


def linux_runtime_files(hlos):
    if hlos != "linux":
        return []
    value = os.environ.get("TILELANG_HEXAGON_LINUX_RPC_LIB_DIR")
    assert value, "Linux preparation requires TILELANG_HEXAGON_LINUX_RPC_LIB_DIR"
    root = Path(value).resolve()
    assert (root / "libcdsprpc.so").is_file(), "Captured Linux BSP FastRPC library is missing"
    return sorted({p.resolve() for p in root.glob("*.so*") if p.is_file()})


def runtime_identity(repo):
    """Bind the native library actually mapped by this fresh Python process."""
    import tilelang
    from tilelang.hexagon.runtime import session

    assert Path(tilelang.__file__).resolve() == repo / "tilelang/__init__.py"
    assert Path(session.__file__).resolve() == repo / "tilelang/hexagon/runtime/session.py"
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1])
        if path.name.startswith(("libtilelang", "libtvm")) and ".so" in path.name:
            assert not fields[-1].endswith(" (deleted)"), "Loaded compiler library was replaced"
            mapped.add(path.resolve())
    natives = [path for path in mapped if path.name == "libtilelang.so"]
    assert len(natives) == 1, ("Expected one actually loaded libtilelang.so", natives)
    native = natives[0]
    return {
        "loaded_libraries": {str(path): sha(path) for path in sorted(mapped)},
        "python": sys.executable,
        "tilelang": str(Path(tilelang.__file__).resolve()),
        "runtime_sources": {str(path): sha(path) for path in sorted((repo / "tilelang/hexagon/runtime").glob("*.py"))},
        "native": str(native),
        "native_sha256": sha(native),
    }


@contextlib.contextmanager
def frozen_builder(build):
    """Rebuild only project text, compare every byte, then reuse the offline ELF."""
    from tilelang.hexagon.runtime import build as module

    expected = {path.name: path.read_bytes() for path in Path(build["project"]).iterdir() if path.is_file()}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()} == build["project_sha256"]

    def use(project, iface, **kwargs):
        assert iface == build["iface"] and kwargs["dsp_arch"] == build["arch"]
        assert {path.name: path.read_bytes() for path in Path(project).iterdir() if path.is_file()} == expected, "Generated project drift"
        for key, digest in build["artifact_sha256"].items():
            assert sha(build["artifacts"][key]) == digest, "SDK artifact drift"
        return module.FastRPCArtifacts(**build["artifacts"])

    with patch.object(module, "build_cmake_fastrpc", use):
        yield


def check_device_environment(build):
    assert (build["arch"], build["hlos"]) in (("v73", "linux"), ("v79", "android"))
    assert os.environ.get("TL_HEXAGON_DEVICE_TEST") == "1", "Set TL_HEXAGON_DEVICE_TEST=1 for explicit device runs"
    assert os.environ.get("TILELANG_HEXAGON_DOMAIN", "3") == "3", "This benchmark uses cDSP domain 3"
    assert os.environ.get("TILELANG_HEXAGON_DEVICE_LOCK"), "Set a shared TILELANG_HEXAGON_DEVICE_LOCK file"
    serial = os.environ.get("TILELANG_HEXAGON_DEVICE") or os.environ.get("ANDROID_SERIAL")
    ssh = os.environ.get("TILELANG_HEXAGON_SSH_HOST")
    if build["hlos"] == "android":
        assert serial and not ssh, "Android requires an explicit device serial and no SSH target"
    else:
        assert ssh or serial, "Linux requires an explicit SSH target or ADB serial"
    return None if ssh else serial


@contextlib.contextmanager
def device_lock():
    # Own one lock across the complete campaign. The nested runtime must not
    # acquire the same flock through another file description.
    name = "TILELANG_HEXAGON_DEVICE_LOCK"
    path = os.environ[name]
    with open(path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        del os.environ[name]
        try:
            yield
        finally:
            os.environ[name] = path


class Drain:
    def __init__(self, sess, path):
        self.errors = []
        fd = os.dup(sess._proc.stdout.fileno())

        def pump():
            try:
                with os.fdopen(fd, "rb", buffering=0) as source, path.open("wb") as target:
                    while block := source.read(4096):
                        assert target.write(block) == len(block)
                        target.flush()
            except BaseException as error:
                self.errors.append(repr(error))

        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()

    def finish(self, sess, out):
        begin = time.monotonic()
        closed_before = getattr(sess, "_closed", True)
        broken_before = getattr(sess, "_broken", False)
        forced, shutdown = [], []
        original_socket = sess._sock

        class ClosingSocket:
            def __getattr__(self, name):
                return getattr(original_socket, name)

            def sendall(self, value):
                try:
                    result = original_socket.sendall(value)
                    shutdown.append({"bytes": value.hex(), "sent": True})
                    return result
                except BaseException as error:
                    shutdown.append({"bytes": value.hex(), "sent": False, "error": repr(error)})
                    raise

        terminate, kill = sess._proc.terminate, sess._proc.kill

        def tracked_terminate():
            forced.append("terminate")
            return terminate()

        def tracked_kill():
            forced.append("kill")
            return kill()

        with (
            patch.object(sess, "_sock", ClosingSocket()),
            patch.object(sess._proc, "terminate", tracked_terminate),
            patch.object(sess._proc, "kill", tracked_kill),
        ):
            sess.close()
        self.thread.join(timeout=5)
        result = {
            "seconds": time.monotonic() - begin,
            "transport_rc": sess._proc.poll(),
            "drain_finished": not self.thread.is_alive(),
            "drain_errors": self.errors,
            "closed_before": closed_before,
            "broken_before": broken_before,
            "forced_process_actions": forced,
            "shutdown_sends": shutdown,
        }
        (out / "EXIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        assert (
            not closed_before
            and not broken_before
            and not forced
            and shutdown == [{"bytes": "02", "sent": True}]
            and result["transport_rc"] == 0
            and result["drain_finished"]
            and not self.errors
        ), result
