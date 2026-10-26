# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Applying a fitted Jacobian lens.

A :class:`JacobianLens` holds the per-layer ``J_l`` matrices produced by
:func:`jlens.fitting.fit`. :meth:`JacobianLens.apply` runs a forward pass and
reads out the requested layers; :meth:`JacobianLens.transport` is the bare
``J_l @ h`` for callers that already have residuals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


class JacobianLens:
    """A fitted Jacobian lens: per-layer ``J_l`` matrices and the readout method.

    Attributes:
        jacobians: ``{layer_index: Tensor[d_model, d_model]}``. Each ``J_l``
            maps the residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. Jacobians are stored as ``dtype`` (default fp16:
        halves file size; entries are usually O(1), and fp16's extra mantissa
        bits beat bf16 here).

        Raises:
            ValueError: If casting a Jacobian to ``dtype`` would turn a
                finite entry non-finite (a magnitude above 65,504 does so
                for the default fp16). Saving anyway would silently corrupt
                the lens, so nothing is written; pass a wider ``dtype``
                such as ``torch.float32`` to store such a lens. Entries
                that are already non-finite are stored as they are: only
                corruption introduced by the cast itself is refused.
        """
        converted: dict[int, torch.Tensor] = {}
        for layer, J in self.jacobians.items():
            cast = J.to(dtype)
            overflowed = torch.isfinite(J) & ~torch.isfinite(cast)
            if bool(overflowed.any()):
                raise ValueError(
                    f"layer {layer}: {int(overflowed.sum())} finite Jacobian "
                    f"entries would become non-finite when stored as {dtype} "
                    f"(largest finite magnitude in the layer is "
                    f"{J.abs().max().item():.6g}); refusing to save a "
                    f"corrupted lens, pass a wider dtype such as torch.float32"
                )
            converted[layer] = cast
        torch.save(
            {
                "J": converted,
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                f"{path} is not a JacobianLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        return cls(
            jacobians=checkpoint["J"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> JacobianLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt subsets into one
        (``n_prompts``-weighted mean of the inputs).

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted_sum = sum(
                lens.jacobians[layer] * lens.n_prompts for lens in lenses
            )
            merged[layer] = weighted_sum / n_total
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J_l @ h``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        J_bar = self.jacobians[layer].to(residual.device)
        return residual @ J_bar.T

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``positions``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers` when ``use_jacobian`` is ``True``.
            positions: Token positions to read out (Python indexing into the
                sequence; negative indices count from the end). ``None`` returns
                every position.
            max_seq_len: Truncate the prompt to this many tokens.
            use_jacobian: If ``False``, skip the ``J_l`` transport (vanilla
                logit-lens baseline).

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at the
            same positions (same shape). ``n_positions`` is ``len(positions)``,
            or the full sequence length when ``positions`` is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model,
                or (with ``use_jacobian``) not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
            )
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            """Residuals at the requested positions: ``[n_positions, d_model]``."""
            full = activations[layer][0]  # [seq_len, d_model]
            return (full if positions is None else full[list(positions)]).float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = select(layer)
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual).float().cpu()

        model_logits = model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits, input_ids
