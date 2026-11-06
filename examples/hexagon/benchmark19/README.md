# Original 19-configuration SA8650 benchmark

Run the original ADD, MUL, SCALE, RMSNorm and GEMM configurations independently
through the standard TileLang compile/call API on Linux SA8650 / Hexagon v73.
This directory does not implement its own FastRPC session or ARM agent.

## Environment and input data

Build and enter the image described in [the Docker guide](../../../docker/hexagon/README.md).
The benchmark uses that image's TileLang, Hexagon SDK, Python packages and ARM64 C
compiler. Use the same board BSP runtime libraries when building the image.
No llama.cpp installation is needed to run the TileLang measurements.

The fixed input/reference tensors are distributed separately as
`benchmark19-inputs-v1.tar.gz`. Obtain this data package from the benchmark
maintainer; it is not downloaded automatically or stored in Git. Extract it into
a writable work directory. Verify its SHA256 against `configs/data_package.json`
before extraction. Its root must contain `map/` and `gemm/`:

```bash
mkdir -p /work/benchmark19-data
tar -xzf /work/benchmark19-inputs-v1.tar.gz -C /work/benchmark19-data
export BENCH19_DATA_DIR=/work/benchmark19-data
cd /opt/tilelang/examples/hexagon/benchmark19
python3 tools/data_files.py --data-dir "$BENCH19_DATA_DIR"
```

`configs/inputs.json` records the expected file sizes, dtypes and SHA256 hashes
for all 19 cases and three seeds. Verification requires no board connection.
Each selected case is verified before compilation or execution. Data from the
historical handoff package can also be used directly by pointing
`BENCH19_DATA_DIR` at its `assets/data` directory.

## Run one case or all cases

Inside the container, configure the board address and refresh its SSH host key
using the Docker image's connection helper:

```bash
export TILELANG_HEXAGON_DEVICE=192.168.1.1:5555
export TILELANG_HEXAGON_SSH_HOST=root@192.168.1.1
source scripts/env.sh
connect-board
bash scripts/run_case.sh --list
bash scripts/run_case.sh ADD_128_4096 run
bash scripts/run_case.sh ADD_128_4096 bench
bash scripts/run_all.sh bench
```

`run` performs one measured execution and checks accuracy. `bench` uses seed 1,
five warmups and 128 measured samples. Select `--seed 2`, `--warmups 5` or
`--samples 128` explicitly when needed. Both modes report Kernel and Full.
The original 19 shapes and tile/worker settings are in `configs/cases.json`;
the four 1x2048 cases are included. No additional control shapes are run.

Results default to `/output/benchmark19/<UTC timestamp>-<unique ID>/` so an
unprivileged image user does not need to write under `/opt/tilelang`.
Use `--output-dir /work/my-results` to change the parent directory. Keep a log
outside the source checkout when sharing a run:

```bash
mkdir -p /output/benchmark19-logs
set -o pipefail
bash scripts/run_all.sh bench 2>&1 \
  | tee "/output/benchmark19-logs/all-$(date -u +%Y%m%d-%H%M%S).log"
```

The runner executes cases serially and stops on the first failure. It never
retries an operator, restarts a service or reboots the board. The local file lock
only coordinates this benchmark's runners for the same user; reserve the board
before running if other users or applications share it.

## Measurement conditions

- Inputs and outputs are distinct. Inputs are fixed and published before the
  repeated DSP measurements; input DMA and waits remain, while repeated input
  cache cleans are omitted. This is a fixed-input, warmed benchmark, not a
  measurement of ARM input updates or model inference.
- Kernel measures the complete generated operator entry, including its internal
  DMA, format conversion/packing, computation, output writes and synchronization.
- Full additionally measures the output cache publication and the nested timer
  reads. It excludes ARM dispatch, RPC, file transfer and input preparation.
- Output publication occurs after every warmup and sample. The benchmark uses
  the historical GGML output cache policy, including a whole data-cache
  flush/invalidate for outputs larger than 128 KiB.
- Both columns come from paired 19.2 MHz DSP qtimer readings; results contain
  raw ticks and median/min/max in microseconds. Timer overhead is not subtracted.
- The benchmark requests compute-client classification, MAX core/bus corners and
  HVX power around the measurement loop, then removes its own votes. Requests,
  queries and cleanup are outside the measured intervals. Return codes and
  before/after core/DMA clock queries are recorded in `power.json`. Bus/DDR
  frequency and sustained clock residency are not independently verified.
- Accuracy uses the original reference files and error thresholds: NMSE below
  1e-7 for map operators and 5e-4 for GEMM. Map tensors are FP32; GEMM uses FP32
  activations, FP16 weights and FP32 output through the original conversion/HMX
  implementation. See each case's recorded source and specification.

Every successful case directory contains `results.json`, `power.json`,
`output.bin`, generated operator/timer source and `status.json`. The top-level
`summary.json` collects successful cases. `--list` and `--help` never contact a
board. Historical run outputs, compiled libraries and archives are not bundled.

## Optional llama.cpp comparison

Install and build llama.cpp separately. The historical reference used
`ggml-org/llama.cpp` at commit
`4fc4ec5541b243957ae5099edb67372f8f3b550e` with Linux/v73 compatibility and DSP
profiling patches plus a custom single-operator runner. Its source checkout,
patches, binaries and build process are not part of this directory.

The unmodified commit does not reproduce the historical two-column output:
Kernel was timed inside native `execute_op`; Full was timed around native
`proc_op_req`, including dispatcher preparation and output cache publication.
TileLang Full does not include the GGML dispatcher, and TileLang repeats inside
one standard call whereas the GGML runner submitted graphs separately. Match
these boundaries, inputs, cache policy and power requests when comparing; the
same commit alone does not establish equivalent timing scope. No automatic
TileLang/llama.cpp ratio is reported by this standalone runner.

## Source and artifact policy

Keep only scripts, the original configuration and input hash manifest in Git.
Do not commit `runs/`, input binaries, generated source, archives, build output,
DSOs, device keys, environment files or developer paths. The external input
package contains tensor data only, without SDK/BSP libraries or native binaries.
New measurements are recorded independently and do not overwrite historical
delivery results.
