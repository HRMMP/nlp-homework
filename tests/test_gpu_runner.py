"""Idle selection must reject allocated GPUs even when utilization is zero."""

from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from scripts.run_on_idle_gpu import idle_candidates, reserve_idle_gpu


def gpu(index, used=15, utilization=0):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "memory_used_mib": used,
        "utilization_percent": utilization,
    }


def test_allocated_memory_is_not_idle():
    assert [
        g["index"] for g in idle_candidates([gpu(0, used=48000), gpu(1)], set())
    ] == [1]


def test_existing_process_excludes_low_memory_gpu():
    assert idle_candidates([gpu(0)], {"GPU-0"}) == []


def test_active_gpu_is_not_idle():
    assert idle_candidates([gpu(0, utilization=90)], set()) == []


def test_selection_is_deterministic_and_empty_is_allowed():
    assert idle_candidates([], set()) == []
    assert [g["index"] for g in idle_candidates([gpu(3), gpu(0)], set())] == [0, 3]


def test_recheck_rejects_gpu_that_becomes_allocated():
    with (
        TemporaryDirectory() as directory,
        patch("scripts.run_on_idle_gpu.tempfile.gettempdir", return_value=directory),
        patch("scripts.run_on_idle_gpu.time.sleep"),
        patch(
            "scripts.run_on_idle_gpu.snapshot",
            side_effect=[
                ([gpu(0)], set()),
                ([gpu(0, used=48000)], set()),
            ],
        ),
        pytest.raises(RuntimeError, match="No GPU"),
    ):
        reserve_idle_gpu(256, 5)
