# Copyright 2026 The jlens Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.digest."""

from jlens.digest import canonical_digest
from jlens.fitting import fit
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


def _fit_lens(seed: int = 0) -> JacobianLens:
    model = TinyDecoder(seed=seed)
    return fit(
        model,
        ["the quick brown fox", "jumps over the lazy dog"],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )


def test_digest_is_deterministic_and_well_formed():
    lens = _fit_lens()
    d1 = canonical_digest(lens)
    d2 = canonical_digest(lens)
    assert d1 == d2
    assert len(d1) == 64 and int(d1, 16) >= 0  # 64 lowercase hex chars


def test_digest_survives_save_load_round_trip(tmp_path):
    lens = _fit_lens()
    before = canonical_digest(lens)
    path = (tmp_path / "lens.pt").as_posix()
    lens.save(path)
    after = canonical_digest(JacobianLens.load(path))
    assert before == after


def test_digest_distinguishes_different_lenses():
    a = canonical_digest(_fit_lens(seed=0))
    b = canonical_digest(_fit_lens(seed=1))
    assert a != b
