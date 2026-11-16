# FlashAttention comparison

This diagnostic compares the current `example_flash_attention.py` Crouton factory
with a matched row-major staged control using the current DSL and runtime.
Both use FP16 HMX outputs and running output buffers, FP32 softmax state,
the same exponential and online updates, and fixed `M=BN=D=64`,
`SEQ=256` or `1024`. Layout conversions
and the row-reduction trees differ; cross-layout bit equality is reported, not
required. Each function must produce identical bits across repeated calls.

Use a Linux host with the repository's Python/native build and local Hexagon SDK.
Preparation also requires `pyelftools` (`python -m pip install pyelftools`).
The commands below run from the repository root. Preparation uses only local
SDK tools and rejects runtime transports and Python network connections.
It never opens a Session on a device.

```bash
python examples/hexagon/flash_attention_bench/probe.py prepare \
  --arch v79 --hlos android --seq 256 --output build/flash-bench/v79-s256
python examples/hexagon/flash_attention_bench/probe.py run \
  --package build/flash-bench/v79-s256
```

`run` without `--run-device` only verifies hashes and regenerates the project
text for an exact comparison. TileLang template headers, sources, the loaded TileLang/TVM native
libraries, runtime code, and SDK products are frozen. Keep the checkout, package
and loaded library paths intact; changed inputs require a fresh package.
`--templates path/to/src --label name` selects an explicit diagnostic header
snapshot without editing the repository templates.

Device execution requires explicit opt-in, a configured target, and a shared
lock file. Use your device's existing environment and an isolated remote
working directory. Run qualification first; `--rounds 6` also includes an initial
qualification and stops on its first failure.

```bash
export TL_HEXAGON_DEVICE_TEST=1
export ANDROID_SERIAL=your-device-serial
export TILELANG_HEXAGON_DEVICE_LOCK=/tmp/tilelang-hexagon-device.lock
python examples/hexagon/flash_attention_bench/probe.py run --run-device \
  --package build/flash-bench/v79-s256 --output build/flash-bench/qualification \
  --rounds 0 --cache 0
python examples/hexagon/flash_attention_bench/probe.py run --run-device \
  --package build/flash-bench/v79-s256 --output build/flash-bench/performance \
  --rounds 6 --cache 0
```

For v73/Linux prepare with `--arch v73 --hlos linux`. Supply the existing Linux
cross compiler and captured BSP libraries through
`TILELANG_HEXAGON_LINUX_CC` and `TILELANG_HEXAGON_LINUX_RPC_LIB_DIR`, together
with any required `TILELANG_HEXAGON_LEGACY_COMPUTE_RES` and
`TILELANG_HEXAGON_ELF_CRC` settings. Preserve these build settings during
verification/run. A Linux device run uses `TILELANG_HEXAGON_SSH_HOST` or an
explicit ADB serial. Preparation and offline verification do not require a
reachable board. Support for preparing this target is not a device-validation
claim.

The measured investigation and rejected optimization candidates are recorded in
[INVESTIGATION.md](INVESTIGATION.md).

## Measurement and checks

Each RPC performs four warmup entries and eight measured entries in balanced
ABBA/BAAB blocks at the same addresses. Six rounds alternate the initial label
and A/A control order, recording Crouton/staged, Crouton/Crouton, and
staged/staged calls. The recorded function address and ELF symbol offsets bind
A/A to the same function. No samples are dropped or corrected using A/A.
`RESULT.json` retains every entry and block/round percentage, including both
global processor-cycle and 19.2 MHz Qtimer counts.

The interval contains the complete generated entry, call/counter overhead and
end `syncht`. It includes normal input copies and HMX/HVX work. Session resource
acquisition/release, RPC, output snapshots, initialization and cache preparation
are outside it. `--cache 0` reads each 32-byte input line before timing;
`--cache 1` requests a range flush/invalidate. These describe procedures, not
proof of physical cache residency.

Every output, including warmups, is checked against a dense FP64 oracle using
fixed seed 75: finite values, maximum absolute error below 0.05, normalized
maximum error below 0.01, and normalized RMS error below 0.01 must all pass.
All DDR allocation and output-snapshot guards, the unused VTCM tail, input bytes
at RPC boundaries, capacity, function bindings and addresses are checked.
There are no internal VTCM guards or per-entry input readbacks. Raw allocations,
full counter records and numerical/bit/ULP differences are saved.

`COMPLETE.json` is written only after a successful shutdown command, transport
exit code zero, complete output drainage and no terminate/kill fallback.
A complete kernel-error reply permits saving raw evidence; a transport failure
never triggers another read RPC. Device recovery after a transport failure is
an operator decision; the tool does not retry automatically.
