"""Numerical qualification for the Flash attention performance comparison."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def protocol():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/flash_attention_bench/protocol.py"
    spec = importlib.util.spec_from_file_location("flash_bench_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seq", [256, 1024])
def test_rounded_reference_qualifies(protocol, seq):
    _, reference = protocol.case(seq)
    output = reference.astype("<f2").tobytes()
    assert protocol.numerical(output, reference)["passed"]


def test_small_absolute_error_does_not_qualify_zero_attention(protocol):
    _, reference = protocol.case(1024)
    output = np.zeros_like(reference, dtype="<f2").tobytes()
    result = protocol.numerical(output, reference)
    # The example's historical absolute threshold accepts this missing output.
    assert result["max_abs_error"] < 0.05
    assert not result["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_attention_does_not_qualify(protocol, value):
    _, reference = protocol.case(256)
    output = reference.astype("<f2")
    output[0, 0] = value
    assert not protocol.numerical(output.tobytes(), reference)["passed"]
