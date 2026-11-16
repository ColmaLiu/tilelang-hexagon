"""Prepare offline, then explicitly run a frozen FlashAttention comparison."""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.dont_write_bytecode = True
import support as safety


def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
p = module(HERE / "protocol.py", "flash_attention_bench_protocol")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def setup(package, arch, hlos, offline):
    assert "tilelang" not in sys.modules, "Use a fresh Python process for each package"
    safety.environment(REPO, arch, hlos, package, offline)


def check_import(package):
    import tilelang
    from tilelang.env import TILELANG_TEMPLATE_PATH

    assert Path(tilelang.__file__).resolve() == REPO / "tilelang/__init__.py"
    assert Path(TILELANG_TEMPLATE_PATH).resolve() == package / "template/src"


def lowered(build):
    from tilelang.engine.param import KernelParam

    return [
        (k["entry"], Path(k["source"]).read_text(), [KernelParam(v["dtype"], v["shape"]) for v in k["params"]], [])
        for k in build["kernels"]
    ]


def dependencies(project, arch, root, sdk):
    directories = [d for d in Path(project).glob(f"hexagon_*_{arch}") if (d / "build.ninja").is_file()]
    assert len(directories) == 1, ("SDK Ninja build directory", directories)
    directory = directories[0]
    commands = subprocess.check_output([sdk.ninja, "-C", str(directory), "-t", "commands"], text=True).splitlines()
    commands = [s for s in commands if "_dsp.cc" in s and " -c " in s]
    assert len(commands) == 1
    command = commands[0]
    tokens = shlex.split(command)
    for flag in ["-O2", "--disable-handle-qfp", "-hexagon-insert=false", "-mhvx-length=128B", "-m" + arch, "-I" + str(root)]:
        assert flag in tokens, flag
    assert ("-enable-xqf-gen=true" in tokens) == (arch == "v79"), "architecture-specific SDK flags"
    deps = subprocess.check_output([sdk.ninja, "-C", str(directory), "-t", "deps"], text=True)
    paths = sorted({str(Path(s.strip()).resolve()) for s in deps.splitlines() if "/tl_templates/hexagon/" in s})
    assert paths and all(Path(s).is_relative_to(root) for s in paths)
    return {"compile_command": command, "headers": {s: sha(s) for s in paths}}


def prepare(args):
    package = args.output.resolve()
    assert not package.exists()
    package.mkdir(parents=True)
    source = args.templates.resolve()
    dest = package / "template/src/tl_templates/hexagon"
    dest.mkdir(parents=True)
    headers = sorted((source / "tl_templates/hexagon").glob("*.h"))
    assert headers
    origin_hashes = {str(path): sha(path) for path in headers}
    for path in headers:
        shutil.copyfile(path, dest / path.name)
    assert all(sha(path) == h == sha(dest / Path(path).name) for path, h in origin_hashes.items())
    setup(package, args.arch, args.hlos, True)
    with safety.offline():
        prepare_offline(args, package, source, dest, origin_hashes)


def prepare_offline(args, package, source, dest, origin_hashes):
    check_import(package)
    from tilelang.hexagon.runtime import session, fastrpc
    from tilelang.hexagon.runtime.env import HexagonSDK
    from tilelang.hexagon.runtime.build import build_cmake_fastrpc
    from tilelang.backend.target import determine_target

    target = determine_target("hexagon", return_object=True)
    factory_path = HERE / "factories.py"
    factory = module(factory_path, "flash_attention_bench_factories")
    funcs = [factory.make_crouton(64, args.seq, 64, 64), factory.make_staged(64, args.seq, 64, 64)]
    identity = safety.runtime_identity(REPO)
    native = identity["native"]
    kinds = ["crouton", "staged"]
    data = []
    kernels = session.lower_kernels([(f, []) for f in funcs], target=target)
    for (entry, src, params, outs), func, kind in zip(kernels, funcs, kinds):
        assert not outs
        sp = package / f"{kind}.cc"
        sp.write_text(src)
        tp = package / f"{kind}.tir"
        tp.write_text(func.script(show_meta=True))
        highs = re.findall(r"tl_vtcm_shared_high_water = (\d+)u;", src)
        data.append(
            {
                "entry": entry,
                "source": str(sp),
                "source_sha256": sha(sp),
                "tir": str(tp),
                "tir_sha256": sha(tp),
                "kind": kind,
                "params": [{"dtype": str(v.dtype), "shape": [int(x) for x in v.shape]} for v in params],
                "out_idx": [],
                "arena_high_values": [int(v) for v in highs],
            }
        )
    # The guard begins at the exact generated entry's allocation boundary.
    high = {}
    for kernel in data:
        values = kernel["arena_high_values"]
        assert len(values) == 1 and values[0] > 0 and values[0] % 128 == 0, (kernel["kind"], values)
        high[kernel["kind"]] = values[0]
    wrapper = (HERE / "wrapper.cc.in").read_text()
    replacements = {
        "SEQ": args.seq,
        "KV_BYTES": args.seq * 64 * 2,
        "HIGH_A": high["crouton"],
        "HIGH_B": high["staged"],
        "MAX_HIGH": max(high.values()),
    }
    for key, value in replacements.items():
        wrapper = wrapper.replace("@" + key + "@", str(value))
    assert not re.search(r"@[A-Z_]+@", wrapper)
    wp = package / "wrapper.cc"
    wp.write_text(wrapper)
    params = [{"dtype": "int32", "shape": []}] * 3 + [
        {"dtype": "float16", "shape": [64, 64]},
        {"dtype": "float16", "shape": [args.seq, 64]},
        {"dtype": "float16", "shape": [args.seq, 64]},
        {"dtype": "float16", "shape": [64, 64]},
        {"dtype": "uint8", "shape": [12 * 8448]},
        {"dtype": "int64", "shape": [12, 40]},
    ]
    data.append({"entry": "flash_compare", "source": str(wp), "source_sha256": sha(wp), "params": params, "out_idx": [], "kind": "caller"})
    record = {
        "repo": str(REPO),
        "head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "native": native,
        "identity": identity,
        "arch": args.arch,
        "hlos": args.hlos,
        "build_environment": safety.build_environment(),
        "target": str(target),
        "seq": args.seq,
        "header_label": args.label,
        "template_origin": str(source),
        "template_origin_sha256": origin_hashes,
        "kernels": data,
        "arena_high": high,
        "arena_required": max(high.values()) + 128,
        "wrapper_index": 2,
        "scope": "Complete generated entry plus syncht. VTCM/HMX already held; reset/cache preparation/snapshot/RPC outside. No altered optimization flags. Crouton/staged rounding trees may differ.",
    }
    project, iface = session.write_project(str(package / "project"), [(n, s, fastrpc._plan(pars, [])) for n, s, pars, _ in lowered(record)])
    files = (
        list(dest.glob("*.h"))
        + list((REPO / "tilelang/hexagon/runtime").glob("*.py"))
        + list(Path(project).iterdir())
        + [
            Path(__file__),
            HERE / "protocol.py",
            HERE / "wrapper.cc.in",
            factory_path,
            HERE / "support.py",
            REPO / "examples/hexagon/example_flash_attention.py",
        ]
    )
    files += safety.linux_runtime_files(args.hlos)
    files += [Path(d[k]) for d in data for k in ["source", "tir"] if k in d]
    frozen = {str(x): sha(x) for x in files if x.is_file()}
    record.update(
        project=project,
        iface=iface,
        project_sha256={x.name: sha(x) for x in Path(project).iterdir() if x.is_file()},
        frozen_inputs=frozen,
    )
    save(package / "PREPARE.json", record)
    artifacts = dataclasses.asdict(
        build_cmake_fastrpc(project, iface, sdk=HexagonSDK(dsp_arch=args.arch), dsp_arch=args.arch, verbose=True)
    )
    assert all(sha(path) == digest for path, digest in frozen.items())
    assert {x.name: sha(x) for x in Path(project).iterdir() if x.is_file()} == record["project_sha256"]
    from elftools.elf.elffile import ELFFile

    with open(artifacts["dsp_skel"], "rb") as stream:
        elf = ELFFile(stream)
        symbols = {
            s.name: {"offset": int(s["st_value"]), "bytes": int(s["st_size"])}
            for s in elf.get_section_by_name(".symtab").iter_symbols()
            if s["st_info"]["type"] == "STT_FUNC"
        }
    offsets = {kind: symbols[f"tl_session_kernel_{i}"]["offset"] for i, kind in enumerate(kinds)}
    assert len(set(offsets.values())) == 2
    tool = Path(HexagonSDK(dsp_arch=args.arch).tools_root) / "bin/hexagon-llvm-objdump"
    (package / "skel.asm").write_bytes(subprocess.check_output([str(tool), "-d", "--demangle", artifacts["dsp_skel"]]))
    record.update(
        artifacts=artifacts,
        artifact_sha256={k: sha(v) for k, v in artifacts.items() if v},
        function_offsets=offsets,
        entry_symbols={k: symbols[f"tl_session_kernel_{i}"] for i, k in enumerate(kinds)},
        template_dependencies=dependencies(project, args.arch, package / "template/src", HexagonSDK(dsp_arch=args.arch)),
        status="BUILT_OFFLINE",
    )
    save(package / "BUILD.json", record)
    print("BUILT_OFFLINE", package, flush=True)


def verify(package):
    build = load(package / "BUILD.json")
    assert all(sha(x) == h for x, h in build["frozen_inputs"].items()), "frozen input drift"
    assert all(sha(build["artifacts"][x]) == h for x, h in build["artifact_sha256"].items()), "artifact drift"
    assert safety.runtime_identity(REPO) == build["identity"]
    assert safety.build_environment() == build["build_environment"], "build environment drift"
    assert build["repo"] == str(REPO)
    return build


def execute(sess, build, args, out):
    import numpy as np

    sizes = p.sizes(build["seq"])
    values, reference = p.case(build["seq"])
    allocations = {k: sess.buffer(n + 256) for k, n in sizes.items()}
    views = {k: b.at(128, sizes[k]) for k, b in allocations.items()}
    for name, b in allocations.items():
        b.write(p.GUARD + values.get(name, bytes(sizes[name])) + p.GUARD)
    for name, v in values.items():
        (out / f"{name}.input.bin").write_bytes(v)
    np.save(out / "REFERENCE.npy", reference)
    schedule = p.schedule(args.rounds)
    save(out / "SCHEDULE.json", schedule)
    records = []
    first = {}
    addresses = None
    from tilelang.hexagon.runtime.session import _KernelError

    for number, job in enumerate(schedule):
        folder = out / f"call-{number:03d}-{job['phase']}-mode{job['mode']}"
        folder.mkdir()
        for name in ["O", "Snapshots", "Records"]:
            views[name].write(bytes(sizes[name]))
        for name, v in values.items():
            assert views[name].read() == v
        save(
            folder / "START.json",
            {
                **job,
                "seq": build["seq"],
                "cache": args.cache,
                "buffers": {k: {"id": b.id, "offset": b.offset, "nbytes": b.nbytes} for k, b in views.items()},
            },
        )

        def capture(folder=folder):
            raw = {k: b.read() for k, b in allocations.items()}
            for k, v in raw.items():
                (folder / f"{k}.allocation.bin").write_bytes(v)
            save(folder / "RAW-HASHES.json", {k: sha(folder / f"{k}.allocation.bin") for k in raw})
            return raw

        try:
            sess.kernels[build["wrapper_index"]](args.cache, job["mode"], job["start"], *[views[k] for k in sizes])
        except _KernelError:
            capture()
            raise
        raw = capture()
        result = p.decode(raw, args.cache, job["mode"], job["start"], build)
        result.update(job)
        snapshots = raw["Snapshots"][128:-128]
        for i, row in enumerate(result["rows"]):
            value = snapshots[i * 8448 + 128 : (i + 1) * 8448 - 128]
            variant = row["variant"]
            first.setdefault(variant, value)
            assert first[variant] == value, "same function output drift"
        if addresses is None:
            addresses = result["same_addresses"]
        assert addresses == result["same_addresses"], "addresses changed between RPCs"
        save(folder / "RESULT.json", result)
        assert result["numeric_passed"]
        records.append(result)
        save(out / "PROGRESS.json", {"rpc_completed": len(records), "entries": len(records) * 12, "rpc_total": len(schedule)})
    result = {
        "records": records,
        "summary": p.summarize(records),
        "seq": build["seq"],
        "header_label": build["header_label"],
        "cache": args.cache,
        "repeated_bits_stable": True,
        "scope": build["scope"],
    }
    save(out / "RESULT.json", result)


def verify_project(build, package):
    from tilelang.hexagon.runtime import session, fastrpc, build as builder

    with tempfile.TemporaryDirectory(prefix="verify-project-", dir=package / "tmp") as directory:
        project, iface = session.write_project(directory, [(n, s, fastrpc._plan(pars, [])) for n, s, pars, _ in lowered(build)])
        with safety.frozen_builder(build):
            artifacts = builder.build_cmake_fastrpc(project, iface, dsp_arch=build["arch"])
        assert dataclasses.asdict(artifacts) == build["artifacts"]


def run(args):
    package = args.package.resolve()
    metadata = load(package / "BUILD.json")
    setup(package, metadata["arch"], metadata["hlos"], not args.run_device)
    with safety.offline():
        check_import(package)
        build = verify(package)
        verify_project(build, package)
    if not args.run_device:
        print("VERIFIED_OFFLINE", package)
        return
    assert args.output is not None, "--output is required with --run-device"
    serial = safety.check_device_environment(build)
    out = args.output.resolve()
    from tilelang.hexagon.runtime import session
    from tilelang.hexagon.runtime.env import HexagonSDK

    with safety.device_lock():
        out.mkdir(parents=True, exist_ok=False)
        save(
            out / "START.json",
            {
                "build": str(package / "BUILD.json"),
                "build_sha256": sha(package / "BUILD.json"),
                "rounds": args.rounds,
                "cache": args.cache,
                "arch": build["arch"],
                "hlos": build["hlos"],
            },
        )
        sess = None
        drain = None
        failure = None
        try:
            with safety.frozen_builder(build):
                sess = session.Session.of_lowered(
                    lowered(build), sdk=HexagonSDK(dsp_arch=build["arch"]), dsp_arch=build["arch"], serial=serial
                )
            drain = safety.Drain(sess, out / "agent.log")
            assert dataclasses.asdict(sess._artifacts) == build["artifacts"]
            save(
                out / "ARTIFACTS.json",
                {"paths": dataclasses.asdict(sess._artifacts), "device_dir": sess._device_dir, "transport_pid": sess._proc.pid},
            )
            execute(sess, build, args, out)
        except BaseException as error:
            failure = error
            save(out / "ERROR.json", {"type": type(error).__name__, "error": str(error)})
        finally:
            if drain:
                try:
                    drain.finish(sess, out)
                except BaseException as error:
                    save(out / "EXIT-ERROR.json", {"error": str(error), "type": type(error).__name__})
                    if failure is None:
                        failure = error
            elif sess is not None:
                sess.close()
        if failure is not None:
            raise failure
        verify(package)
        save(out / "COMPLETE.json", {"strict_exit": True, "result_sha256": sha(out / "RESULT.json")})
        print("COMPLETE", out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--seq", type=int, choices=[256, 1024], required=True)
    q.add_argument("--arch", choices=["v73", "v79"], required=True)
    q.add_argument("--hlos", choices=["linux", "android"], required=True)
    q.add_argument("--templates", type=Path, default=REPO / "src")
    q.add_argument("--label", default="upstream")
    q = sub.add_parser("run")
    q.add_argument("--package", type=Path, required=True)
    q.add_argument("--output", type=Path)
    q.add_argument("--rounds", type=int, choices=[0, 6], default=0)
    q.add_argument("--cache", type=int, choices=[0, 1], default=0)
    q.add_argument("--run-device", action="store_true")
    args = parser.parse_args()
    assert not sys.flags.optimize, "Python -O disables benchmark checks"
    if args.command == "prepare" and (args.arch, args.hlos) not in [("v73", "linux"), ("v79", "android")]:
        parser.error("Use v73/linux or v79/android")
    if args.command == "run" and args.run_device and args.output is None:
        parser.error("--output is required with --run-device")
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
