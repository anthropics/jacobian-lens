# Copyright 2026 The jlens Authors
# SPDX-License-Identifier: Apache-2.0
"""Content digest for fitted Jacobian lenses.

A fitted :class:`~jlens.lens.JacobianLens` is an artifact worth referencing
stably: pre-fitted lenses are shared across machines and runs, and downstream
tooling may need to verify that two parties are using the same lens. File
hashes are fragile for this — re-saving the same lens produces a different
file. :func:`canonical_digest` instead hashes the lens *content*: the fp16
bytes of each Jacobian matrix in ascending layer order. Two lenses with
identical matrices produce identical digests regardless of how, whether, or
how many times they were serialized.
"""

from __future__ import annotations

import hashlib

import torch

from jlens.lens import JacobianLens

__all__ = ["canonical_digest"]

# Matches the dtype used by JacobianLens.save, so a lens has the same digest
# before and after a save/load round trip.
_DIGEST_DTYPE = torch.float16


def canonical_digest(lens: JacobianLens) -> str:
    """Return a SHA-256 hex digest of the lens content.

    The digest covers, in ascending layer order, the layer index followed by
    the fp16 bytes of that layer's Jacobian matrix. It is independent of
    serialization: freshly fitted, saved, reloaded, or merged-and-resaved
    lenses with identical matrices hash identically.

    Args:
        lens: A fitted Jacobian lens.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    h = hashlib.sha256()
    for layer in lens.source_layers:
        h.update(str(layer).encode())
        h.update(
            lens.jacobians[layer]
            .to(_DIGEST_DTYPE)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )
    return h.hexdigest()
