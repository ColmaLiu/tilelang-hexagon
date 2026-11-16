"""Pure host protocol for the current SEQ256/1024 flash experiment.

No TileLang, SDK or device imports. All sizes below are payload sizes; every
resident allocation has an additional 128-byte guard at each end.

Predeclared numerical gate for this experiment: finite outputs, max absolute
error < 0.05, max error / reference maximum < 0.01, and RMS error / reference
RMS < 0.01. This does not redefine the shipped example's numerical contract.
"""

import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

GUARD = b"\xa5" * 128
M = D = BN = 64
SEQUENCES = (256, 1024)
SEED = 75
ENTRIES = 12
WARMUP_ENTRIES = 4
SNAP_BYTES = 8448
WORDS = 40
MAGIC = 0x750116
KINDS = ["crouton", "staged"]
CACHES = ["read-prepared", "range-flush-invalidated"]
LABELS = (0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1)
BLOCKS = ("warmup_ABBA", "measured_ABBA", "measured_BAAB")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sizes(seq):
    assert seq in SEQUENCES, ("sequence", seq)
    return {
        "Q": M * D * 2,
        "K": seq * D * 2,
        "V": seq * D * 2,
        "O": M * D * 2,
        "Snapshots": ENTRIES * SNAP_BYTES,
        "Records": ENTRIES * WORDS * 8,
    }


def case(seq):
    sizes(seq)
    rng = np.random.default_rng(SEED)
    q = (rng.standard_normal((M, D)) * 0.3).astype("<f2")
    k = (rng.standard_normal((seq, D)) * 0.3).astype("<f2")
    v = (rng.standard_normal((seq, D)) * 0.3).astype("<f2")
    scores = q.astype(np.float64) @ k.astype(np.float64).T
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    reference = (weights @ v.astype(np.float64)) / weights.sum(axis=1, keepdims=True)
    return {"Q": q.tobytes(), "K": k.tobytes(), "V": v.tobytes()}, reference


def checksum(inputs):
    # The caller reads one uint32 per32-byte line in Q,K,V order outside timing.
    return sum(int(np.frombuffer(inputs[name], "<u4")[::8].sum(dtype=np.uint64)) for name in ("Q", "K", "V")) & 0xFFFFFFFF


def numerical(raw, reference):
    output = np.frombuffer(raw, "<f2").reshape(M, D).astype(np.float64)
    finite = bool(np.isfinite(output).all() and np.isfinite(reference).all())
    if not finite:
        return {
            "finite": False,
            "max_abs_error": None,
            "rms_error": None,
            "relative_max_error": None,
            "relative_rms_error": None,
            "threshold": 0.05,
            "relative_max_threshold": 0.01,
            "relative_rms_threshold": 0.01,
            "passed": False,
        }
    error = output - reference
    maximum = float(np.abs(error).max())
    rms = float(np.sqrt(np.mean(error * error)))
    ref_max = float(np.abs(reference).max())
    ref_rms = float(np.sqrt(np.mean(reference * reference)))
    relative_max = maximum / ref_max if ref_max else (0.0 if maximum == 0 else None)
    relative_rms = rms / ref_rms if ref_rms else (0.0 if rms == 0 else None)
    passed = maximum < 0.05 and relative_max is not None and relative_max < 0.01 and relative_rms is not None and relative_rms < 0.01
    return {
        "finite": True,
        "max_abs_error": maximum,
        "rms_error": rms,
        "reference_max_abs": ref_max,
        "reference_rms": ref_rms,
        "relative_max_error": relative_max,
        "relative_rms_error": relative_rms,
        "threshold": 0.05,
        "relative_max_threshold": 0.01,
        "relative_rms_threshold": 0.01,
        "passed": bool(passed),
    }


def payloads(raw, seq):
    expected = sizes(seq)
    assert set(raw) == set(expected), "allocation names"
    for name, size in expected.items():
        value = raw[name]
        assert len(value) == size + 256, (name, "allocation length")
        assert value[:128] == GUARD and value[-128:] == GUARD, (name, "allocation guards")
    return {name: value[128:-128] for name, value in raw.items()}


def decode(raw, cache, mode, start, build):
    assert cache in (0, 1) and mode in (0, 1, 2) and start in (0, 1)
    seq = int(build["seq"])
    p = payloads(raw, seq)
    inputs, reference = case(seq)
    for name, value in inputs.items():
        assert p[name] == value, (name, "input mutation")
    highs = build["arena_high"]
    offsets = build["function_offsets"]
    required = int(build["arena_required"])
    assert set(highs) == set(offsets) == set(KINDS)
    assert all(int(highs[kind]) > 0 for kind in KINDS)
    assert required >= max(map(int, highs.values())) + 128
    # Preserve raw unsigned counter bits. Negative status values cannot masquerade
    # as successful metadata, and deltas must fit the bounded flash invocation.
    words = np.frombuffer(p["Records"], "<u8").reshape(ENTRIES, WORDS)
    rows, outputs, addresses, bases, function_addresses = [], [], set(), set(), {}
    for i, values in enumerate(words):
        r = [int(value) for value in values]
        logical = LABELS[i] ^ start
        variant = logical if mode == 0 else (0 if mode == 1 else 1)
        kind = KINDS[variant]
        expected = {
            0: MAGIC,
            1: seq,
            2: 1,
            3: cache,
            4: mode,
            5: start,
            6: i,
            7: logical,
            8: variant,
            9: 0,
            15: int(highs[kind]),
            20: 1,
            21: 1,
            22: 1,
            23: 0,
            26: 8,
            27: M * D * 2,
            28: 0,
            29: checksum(inputs) if cache == 0 else 0,
            30: 0,
            31: 0,
            32: 0,
            33: int(i < WARMUP_ENTRIES),
        }
        for slot, value in expected.items():
            assert r[slot] == value, (i, slot, r[slot], value)
        assert not any(r[34:]), ("reserved records", i)
        assert 0 < r[10] < (1 << 56) and 0 < r[11] < (1 << 63), ("counter delta", i)
        assert r[13] > 0 and r[13] % 2048 == 0 and r[14] >= required, ("VTCM grant", i)
        assert r[24] > 0 and r[25] > 0 and ((r[25] >> 8) & 255) > 0, ("one-worker capacity", i)
        assert all(value > 0 and value % 128 == 0 for value in r[16:20]), ("DDR alignment", i)
        bases.add(r[12] - int(offsets[kind]))
        addresses.add(tuple(r[index] for index in (13, 14, 16, 17, 18, 19)))
        function_addresses.setdefault(kind, set()).add(r[12])
        snapshot = p["Snapshots"][i * SNAP_BYTES : (i + 1) * SNAP_BYTES]
        assert snapshot[:128] == GUARD and snapshot[-128:] == GUARD, ("snapshot guards", i)
        output = snapshot[128:-128]
        outputs.append(output)
        order = "ABBA" if [LABELS[j] ^ start for j in range(i // 4 * 4, i // 4 * 4 + 4)] == [0, 1, 1, 0] else "BAAB"
        rows.append(
            {
                "position": i,
                "seq": seq,
                "cache": cache,
                "warmup": i < WARMUP_ENTRIES,
                "block": BLOCKS[i // 4],
                "order": order,
                "logical": logical,
                "variant": variant,
                "function": kind,
                "ticks": r[10],
                "cycles": r[11],
                "nanoseconds": r[10] * (1000 / 19.2),
                "cycles_per_tick": r[11] / r[10],
                "output_sha256": hashlib.sha256(output).hexdigest(),
                "numeric": numerical(output, reference),
                "raw_words": r,
            }
        )
    assert len(bases) == 1 and min(bases) > 0, "ELF function offset binding"
    assert len(addresses) == 1, "same arena and IO addresses"
    assert all(len(values) == 1 for values in function_addresses.values()), "same function address"
    assert p["O"] == outputs[-1], "final output versus final snapshot"
    comparisons = []
    for i in range(ENTRIES):
        for j in range(i + 1, ENTRIES):
            a = np.frombuffer(outputs[i], "<u2")
            b = np.frombuffer(outputs[j], "<u2")
            changed = a != b
            same = rows[i]["variant"] == rows[j]["variant"]
            assert not same or not changed.any(), ("same-function output bits changed", i, j)
            x, y = a.view("<f2").astype(np.float64), b.view("<f2").astype(np.float64)
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            finite_changed = changed & np.isfinite(x) & np.isfinite(y)
            # Monotonic finite half ordering; signed zeros have distance zero.
            ordered_a = np.where(a & 0x8000, 0x8000 - (a.astype(np.int64) & 0x7FFF), 0x8000 + a.astype(np.int64))
            ordered_b = np.where(b & 0x8000, 0x8000 - (b.astype(np.int64) & 0x7FFF), 0x8000 + b.astype(np.int64))
            ulps = np.abs(ordered_a - ordered_b)[finite_changed]
            ulp = {
                name: float(np.percentile(ulps, percentile)) if ulps.size else 0.0
                for name, percentile in [("median", 50), ("p90", 90), ("p99", 99), ("max", 100)]
            }
            comparisons.append(
                {
                    "positions": [i, j],
                    "same_function": same,
                    "different_elements": int(changed.sum()),
                    "finite_different_elements": int(finite_changed.sum()),
                    "ulp_over_finite_changed": ulp,
                    "max_abs": float(np.abs(x - y).max()) if finite else None,
                }
            )
    assert all(row["numeric"]["passed"] for row in rows), (
        "FP64 numeric gate",
        [(r["position"], r["numeric"]) for r in rows if not r["numeric"]["passed"]],
    )
    blocks = [
        {
            "block": BLOCKS[b],
            "order": rows[4 * b]["order"],
            "positions": list(range(4 * b, 4 * b + 4)),
            "logical": [r["logical"] for r in rows[4 * b : 4 * b + 4]],
            "variants": [r["variant"] for r in rows[4 * b : 4 * b + 4]],
            "ticks": [r["ticks"] for r in rows[4 * b : 4 * b + 4]],
            "cycles": [r["cycles"] for r in rows[4 * b : 4 * b + 4]],
        }
        for b in range(3)
    ]
    return {
        "rows": rows,
        "comparisons": comparisons,
        "block_timings": blocks,
        "numeric_passed": True,
        "same_function_bits_stable": True,
        "same_addresses": list(addresses.pop()),
        "module_bias": bases.pop(),
        "function_addresses": {kind: next(iter(values)) for kind, values in function_addresses.items()},
    }


def schedule(rounds):
    assert rounds in (0, 6), "one qualification RPC or six predeclared rounds"
    result = [{"phase": "qualification", "trial": -1, "mode": 0, "start": 0}]
    for trial in range(rounds):
        for mode in [0, 1, 2] if trial % 2 == 0 else [0, 2, 1]:
            result.append({"phase": "performance", "trial": trial, "mode": mode, "start": trial % 2})
    return result


def _timing_summary(rows):
    groups = {label: [row for row in rows if row["logical"] == label] for label in (0, 1)}
    assert len(groups[0]) == len(groups[1]) and groups[0]
    medians = {
        str(label): {metric: statistics.median(row[metric] for row in group) for metric in ("ticks", "cycles")}
        for label, group in groups.items()
    }
    return {
        "label0": medians["0"],
        "label1": medians["1"],
        "change_percent": {metric: (medians["1"][metric] / medians["0"][metric] - 1) * 100 for metric in ("ticks", "cycles")},
        "positions": {str(label): [row["position"] for row in group] for label, group in groups.items()},
    }


def summarize(records):
    assert records
    signatures, addresses, biases = {}, set(), set()
    result = []
    for record in records:
        rows = record["rows"]
        assert len(rows) == ENTRIES
        addresses.add(tuple(record["same_addresses"]))
        biases.add(record["module_bias"])
        for i, row in enumerate(rows):
            assert row["position"] == i and row["warmup"] is (i < WARMUP_ENTRIES)
            logical = LABELS[i] ^ record["start"]
            variant = logical if record["mode"] == 0 else (0 if record["mode"] == 1 else 1)
            assert row["logical"] == logical and row["variant"] == variant
            assert row["function"] == KINDS[variant] and row["numeric"]["passed"]
            signatures.setdefault((row["seq"], row["function"]), set()).add(row["output_sha256"])
        if record["phase"] != "performance":
            assert record["phase"] == "qualification"
            continue
        measured = rows[WARMUP_ENTRIES:]
        summary = _timing_summary(measured)
        blocks = []
        for b in (1, 2):
            br = rows[4 * b : 4 * b + 4]
            blocks.append({"block": BLOCKS[b], "order": br[0]["order"], **_timing_summary(br)})
        result.append(
            {
                "trial": record["trial"],
                "mode": record["mode"],
                "start": record["start"],
                "warmup_positions": list(range(WARMUP_ENTRIES)),
                "measured_positions": summary.pop("positions"),
                **summary,
                "measured_blocks": blocks,
                "interpretation": "staged versus crouton" if record["mode"] == 0 else "same-function positional A/A",
            }
        )
    assert len(addresses) == len(biases) == 1, "cross-RPC address/ELF binding changed"
    assert all(len(values) == 1 for values in signatures.values()), "cross-RPC same-function output bits changed"
    return result
