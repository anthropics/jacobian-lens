# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""JacobianLens.save() must not silently turn finite entries into inf."""

import os

import pytest
import torch

from jlens import JacobianLens


def _lens(value: float) -> JacobianLens:
    return JacobianLens(
        {0: torch.full((2, 2), value, dtype=torch.float32)}, n_prompts=1, d_model=2
    )


def test_save_rejects_values_that_overflow_fp16(tmp_path):
    path = tmp_path / "lens.pt"
    with pytest.raises(ValueError, match="non-finite"):
        _lens(1e16).save(str(path))
    assert not os.path.exists(path)


def test_save_rejects_values_above_the_fp16_max(tmp_path):
    with pytest.raises(ValueError, match="layer 0"):
        _lens(70_000.0).save(str(tmp_path / "lens.pt"))


def test_save_with_a_wider_dtype_preserves_large_values(tmp_path):
    path = str(tmp_path / "lens.pt")
    _lens(1e16).save(path, dtype=torch.float32)
    loaded = JacobianLens.load(path)
    assert bool(torch.isfinite(loaded.jacobians[0]).all())
    torch.testing.assert_close(loaded.jacobians[0], torch.full((2, 2), 1e16))
    assert bool(torch.isfinite(loaded.transport(torch.ones(2), 0)).all())


def test_save_default_dtype_round_trips_representable_values(tmp_path):
    path = str(tmp_path / "lens.pt")
    _lens(0.5).save(path)
    loaded = JacobianLens.load(path)
    torch.testing.assert_close(loaded.jacobians[0], torch.full((2, 2), 0.5))


def test_save_stores_already_nonfinite_entries_as_they_are(tmp_path):
    path = str(tmp_path / "lens.pt")
    lens = JacobianLens(
        {0: torch.tensor([[float("inf"), 1.0], [1.0, 1.0]])}, n_prompts=1, d_model=2
    )
    lens.save(path)
    loaded = JacobianLens.load(path)
    assert bool(torch.isinf(loaded.jacobians[0][0, 0]))
