# Issue 75: current Flash attention evidence

Investigation draft for [issue #75](https://github.com/xwhzz/tilelang-hexagon/issues/75).
The historical post-PR76 layout ranking is not reproduced on the current baseline. Vector initialization and the Crouton example are retained; none of the screened changes improves both numerical contexts. This investigation does not establish readiness to close the issue while v73 hardware validation is pending.

## Scope and historical reassessment

The baseline is `1ac61170aecbc4b63b72a2f30e28f28cd7713492`, using SDK 6.6.0.0 / Hexagon Tools 19.0.07 and normal ReleaseG `-O2` options on v79.
The workload is one worker, M=BN=D=64, SEQ=256 or 1024, with the example's existing unscaled `softmax(QKᵀ)V` definition.
The staged control uses the current backend and preserves the same direct Q copy; it is not a replay of the entire historical staged binary.

The historical issue recorded approximately 8–9% staged improvements after the change associated with #25, while Crouton changed by approximately +0.5%.
Those issue numbers refer to issues; the implementations were [PR #70](https://github.com/xwhzz/tilelang-hexagon/pull/70) and [PR #76](https://github.com/xwhzz/tilelang-hexagon/pull/76).
PR #70 changed physical traversal, broadcast/reduction structure and the accumulator layout as well as removing conversions.
Before PR #76, initialization used a 32-iteration loop with **two scalar word stores per iteration**: 64 words / 256 bytes. PR #76 replaced that loop with two vector stores.
It also flattened the old TIR output-tile loop. That lowering has since been replaced by the inline C++ HMX driver and its nested tile loops.
Consequently, restoring the historical flattened loop is not a minimal current-backend experiment.

The historical timings establish different performance for different compiled programs; they do not establish an instruction-cache cause.
The issue does not preserve the corresponding ELF artifacts, timer boundaries or A/A distributions needed to reproduce that attribution directly.
Deleting the configuration fill is also not a valid semantic control without proving initialization and VTCM lifetime for every subsequent read.
The experiments below retain complete, correct configuration initialization.

## Measurement and correctness contract

Eighteen completed formal campaigns retain **342 RPCs and 4,104 complete outputs**, including their qualification RPCs and fixed warmups.
Each campaign contains one qualification RPC followed by six predeclared rounds of AB and both same-function controls, AA and BB.
Every RPC has four fixed warmup entries followed by balanced ABBA and BAAB blocks; the starting label alternates by round.
All samples and both starting orders are retained. No outlier filtering, repeated-run selection, empty-timer subtraction or A/A correction is applied.
Reported percentages are medians of the six per-round ratios, not ratios of independently pooled medians.

The caller times the complete normal generated entry, call/argument instrumentation and final `syncht` while VTCM/HMX resources are already held.
Host allocation and input/cache preparation, arena/output reset, guards, snapshots, resource acquisition and the outer FastRPC call are outside this interval.
Normal generated-entry DDR-to-VTCM input copies remain inside the interval.
C15:14 records processor cycles; C31:30 records the wrapping 56-bit 19.2 MHz Qtimer. The latter also advances while the DSP is clock-gated.
Neither is an executed-packet or stall-event counter. Power votes do not establish locked frequency.
All completed campaigns here use read preparation: one word per 32-byte Q/K/V input line, in fixed order.
This specifies preparation rather than physical cache residency; no cold-DRAM or I-cache-flush claim is made.

Inputs are NumPy `default_rng(75)` normal samples scaled by 0.3 and rounded to FP16. An independent FP64 stable-softmax reference starts from their exact bytes.
Before device measurements, the gates were fixed to finite outputs, maximum absolute error <0.05, normalized maximum error <1%, and normalized RMS error <1%.
The normalized denominators are the reference maximum absolute value and reference RMS, respectively.
Every output, including qualification and warmups, passes; same-function repeats are bitwise stable within and across RPCs.
For these particular fixtures, both layouts and all measured controls also return identical bits; that is an observation, not a general equivalence guarantee.
Normalized maximum/RMS errors are 0.09194%/0.05602% at SEQ256 and 0.12186%/0.08051% at SEQ1024.

Independent readers reconstruct records and outputs from raw allocations, rather than accepting the runner's numerical summary.
They verify full input bytes at RPC completion, DDR/snapshot guards, recorded VTCM-tail guards, ELF/function/address bindings, frozen sources/artifacts and strict normal shutdown.
The runner also checks inputs before each RPC. There are no per-entry input snapshots or sentinels around every internal VTCM workspace.

## Current layout comparisons

The PR #259 column uses a separately frozen overlay from `4b9620c5399eadef5cec32b0579e0f4e72aae806`.
[PR #259](https://github.com/xwhzz/tilelang-hexagon/pull/259) remains a diagnostic numerical-fix dependency here; it is not silently integrated into the upstream baseline.
Within each row, both layouts use the same header snapshot and module. Positive values mean staged takes longer than Crouton.
Brackets contain all six round extrema.

| Headers / configuration fill | SEQ | Staged/Crouton cycles change | Qtimer change |
|---|---:|---:|---:|
| Upstream / vector | 256 | +3.6069% [3.4035, 3.7845] | +3.6339% |
| Upstream / vector | 1024 | +4.9585% [4.9015, 5.0119] | +4.9680% |
| Upstream / scalar source | 256 | +1.4942% [1.2466, 1.7162] | +1.5416% |
| Upstream / scalar source | 1024 | +2.2148% [2.1573, 2.3336] | +2.2218% |
| PR #259 / vector | 256 | −1.3608% [−1.4713, −1.1435] | −1.2945% |
| PR #259 / vector | 1024 | −0.9236% [−0.9708, −0.8784] | −0.9106% |
| PR #259 / scalar source | 256 | −2.9611% [−3.0857, −2.7982] | −2.9308% |
| PR #259 / scalar source | 1024 | −2.5082% [−2.5325, −2.4714] | −2.5280% |

The current upstream ranking therefore differs from the historical post-PR76 ranking; the separately fixed headers reverse it again.
Both start groups retain each listed direction, but SEQ256 start-group medians differ by approximately 0.24–0.45 percentage points.
The static scalar-source control changes the actual entry code and register allocation as well as the fill instructions.
It cannot isolate fill latency by subtracting the absolute times of different modules or Sessions.

## Same-entry runtime fill control

A separate diagnostic module selects vector or scalar fill through one volatile selector, written outside the timed interval.
Both labels call the same entry address and compiled body; AA fixes vector selection and BB fixes scalar selection.
The selector is observed after timing, and every output must be bitwise equal across both strategies and all RPCs.
Actual linked code retains both complete initializers: two vector stores or 32 iterations of a two-scalar-store packet.
Both paths initialize the same 256-byte configuration and join the same HMX computation. The numerical headers are the PR #259 snapshot.

| SEQ256 layout | Scalar/vector cycles change, six-round range | Paired cycle difference, median [range] | Qtimer change |
|---|---:|---:|---:|
| Crouton | +2.3038% [2.2918, 2.3141] | +1049.5 [1044, 1054] | +2.3551% |
| Staged | +2.4282% [2.3885, 2.4453] | +1120 [1102, 1128] | +2.3866% |

Both start groups and every measured AB block retain this direction.
This demonstrates a cost of the runtime scalar path in a common compiled program, including selector/branch and downstream execution effects.
There are eight configuration invocations at SEQ256; dividing the full-entry delta by eight would not turn it into isolated store latency.
These results also do not mechanically partition the earlier cross-module difference into fill cost and register-allocation cost.

## Outlining controls: no production recommendation

Normal-SDK SEQ256 controls outline either ordinary HMX GEMM or MAX/SUM reductions without changing their mathematical source bodies.
HMX outlining emits one constant-specialized 188-byte helper shared by both entries; it is not a general runtime-shape helper.
Reduction outlining emits four concrete helpers, retaining the original Crouton and row-major trees.

| Outline / headers | Staged/Crouton cycle change [six-round range] | Crouton absolute change vs matching original module | Staged absolute change |
|---|---:|---:|---:|
| HMX / upstream | +1.4825% [1.2865, 1.6708] | +1.1168% | −0.9662% |
| HMX / PR #259 | +0.2026% [0.0444, 0.3597] | +3.3999% | +5.0214% |
| Reductions / upstream | +5.7772% [5.5037, 5.9148] | −0.0629% | +1.9783% |
| Reductions / PR #259 | +3.2695% [3.0994, 3.4660] | +0.9754% | +5.7018% |

The absolute columns compare different modules/Sessions and are descriptive, not isolated call-cost measurements.
The PR #259 HMX layout contrast is near parity: start-group medians are +0.3445% and +0.0628%, and some Qtimer comparisons are zero.
No outlining control provides a generally faster current implementation; smaller callers alone are not sufficient evidence of improvement.

## Other screened changes

Both candidates below compare original and candidate entries in one module, with the same addresses and a hard requirement for bit-identical outputs across variants and RPCs.
The accumulation candidate moves rescaling after the second GEMM and fuses the two output passes, preserving the intermediate FP16 rounding.
The local-fill candidate keeps both vector stores but materializes their values in one inline assembly block with one temporary vector register.
Neither changes the attention definition or drops initialization.

| Candidate / headers | Candidate/original cycles change, six-round range | Qtimer change |
|---|---:|---:|
| Fused accumulation / upstream | −2.7060% [−2.9211, −2.5521] | −2.7215% |
| Fused accumulation / PR #259 | +0.8169% [0.6778, 0.9823] | +0.7995% |
| Local vector fill / upstream | −1.0652% [−1.1763, −0.7866] | −0.9777% |
| Local vector fill / PR #259 | +4.3552% [4.1690, 4.5239] | +4.2438% |

Both start groups retain each direction. These candidates are rejected because their upstream benefit does not transfer to the separately fixed numerical implementation.
The local-fill assembly uses one scratch register at each initialization site, yet the complete entry's stack frame grows in both contexts.
A local lifetime constraint alone therefore does not establish lower whole-kernel register pressure.
Fusion likewise changes scheduling and register allocation; preserving its arithmetic does not preserve its performance.

## What the actual code explains

Generated C/TIR match across the four header/fill variants at each SEQ, and actual normal SDK flags match.
Nevertheless, entry bodies, stack frames, spills and packet schedules change. The hot code itself changes, not only its placement.
The upstream vector entries contain 8/7 traced QF producer→stack→reload→QF-consumer chains for Crouton/staged; scalar-source entries contain 9/7.
The PR #259 entries have zero such direct sites in the packet-aware straight-basic-block scan.
This is evidence of exposed numerical state in the original compilation; no output error is observed for this fixed Flash input.
The scan is not a complete cross-CFG or whole-library proof, and outlining upstream reductions merely moves remaining QF chains into their callees.

The proposed explanation that the fix newly introduced identity additions is contradicted by the actual SUM bodies.
Both versions contain 95 additions, 32 half-pair QF multiplies and 159 QF→SF conversions in the same 32-row tree body.
That body's static packet count changes from 179 to 344, while instruction words change from 439 to 429.
The whole KV region changes from 872 to 1042 static packets: almost all of that packet-count difference occurs in SUM.
These are unexpanded static regions, not dynamic executed packets, measured cycles or a quantitative attribution of all slowdown.
The evidence supports changed bundling/scheduling and register allocation; it does not justify deleting identity arithmetic or changing numerical semantics.

## Variation retained and remaining work

Across these campaigns, whole-round same-function cycle changes remain within about ±0.073%, but individual blocks and samples are less quiet.
Retained examples include upstream SEQ256's +11.9345% staged sample and +9.9763% paired block; no sample was discarded.
AA blocks include +1.6323% and +1.5934%; reduction-outline upstream retains −1.5150% AA and −2.1465% BB blocks.
Runtime-fill staged retains a +5.6433% scalar sample and a −2.7443% BB block. The local-fill campaigns retain +6.3227% and +8.9772% samples and a +4.5134% BB block. Raw records preserve the corresponding Qtimer values and all smaller deviations.
These observations bound precision and do not constitute a guarantee of universally stable timing.

No controlled same-code relocation experiment or hardware event evidence establishes an instruction-cache or bank-conflict root cause.
This investigation makes no claim that unrelated edits can never affect performance.
Normal SDK v73/Linux builds pass for both SEQ values and both header snapshots. Their DSP ELFs target v73, their agents target AArch64 Linux, and the generated entry retains the IEEE arithmetic attribute.
The repository benchmark also passes v73 offline preparation and project regeneration; its DSP allocated sections, including entry code and relocations, match the original v73 diagnostic exactly.
v73 board correctness and performance remain pending hardware recovery; v79 timing and numerical behavior are not extrapolated to it.
The separate [PR #256](https://github.com/xwhzz/tilelang-hexagon/pull/256) DMA/pipeline implementation is not measured in these eighteen campaigns.
Its integration would require new final-module measurements rather than inheriting the percentages above.

The accompanying benchmark makes the matched layout comparison, numerical gates and raw timing records reproducible. It does not promise performance invariance under arbitrary source edits.
No production optimization is selected from these experiments, and the numerical fix in PR #259 remains a separate draft dependency.
The remaining board run must validate the actual v73 artifacts, correctness and timing on recovered hardware before drawing a cross-device conclusion.

## Repository benchmark revalidation

Four additional campaigns execute the repository version of this benchmark: **76 RPCs and 912 outputs**, separately from the eighteen diagnostic campaigns above.
The generated C/TIR and wrapper are byte-identical to their original counterparts. All allocated DSP ELF sections, including code, GOT and dynamic relocations, also match exactly.
The repeated layout differences retain their direction in both clocks, both starting orders and every AB block:

| Header | SEQ | Staged/Crouton cycles change, six-round range |
|---|---:|---:|
| Upstream | 256 | +3.5778% [3.3922, 3.7656] |
| Upstream | 1024 | +4.9639% [4.9046, 5.0111] |
| PR #259 | 256 | −1.3747% [−1.4584, −1.1195] |
| PR #259 | 1024 | −0.9439% [−0.9767, −0.8669] |

Every numerical, guard, function-binding and strict shutdown check passes. Outputs match the corresponding original fixtures bit for bit.
All new raw samples are retained. The largest absolute A/A or B/B whole-round cycle difference is 0.01481%; the largest block difference is 0.47548%.
These reruns validate the tool migration and the repeated direction of the current layout result; they do not broaden the fixed-input correctness claim.
