"""One-forward comparative readouts with per-sample resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.security.data import render_security_prompt, sha256_file
from jlens.security.runtime import atomic_write_json

SECURITY_CONCEPTS = [
    "injection",
    "inject",
    "malicious",
    "suspicious",
    "unsafe",
    "attack",
    "fake",
    "fraud",
    "trick",
    "ignore",
    "override",
    "instruction",
    "hidden",
    "secret",
    "send",
    "delete",
]
METHODS = ("jlens", "logit_lens", "model_final")
PROMPT_TEMPLATE_VERSION = "security_external_content_v1"


def build_concept_tokens(tokenizer: Any) -> dict[int, dict[str, str]]:
    tokens: dict[int, dict[str, str]] = {}
    for concept in SECURITY_CONCEPTS:
        for variant in (
            concept,
            f" {concept}",
            concept.capitalize(),
            f" {concept.capitalize()}",
        ):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            tokens.setdefault(
                token_id,
                {
                    "concept": concept,
                    "token_text": tokenizer.decode([token_id]),
                },
            )
    if not tokens:
        raise ValueError("No single-token security concepts for this tokenizer")
    return tokens


def top_tokens(
    logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    k = min(top_k, int(logits.numel()))
    values, indices = logits.topk(k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for token_id, value in zip(
            indices.tolist(),
            values.tolist(),
            strict=True,
        )
    ]


def score_concepts(
    logits: torch.Tensor,
    concept_tokens: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Score concepts with one vocabulary sort instead of 53 full scans."""
    vector = logits.detach().float().cpu().flatten()
    token_ids = list(concept_tokens)
    id_tensor = torch.tensor(token_ids, dtype=torch.long)
    scores = vector[id_tensor]
    descending = torch.sort(vector, descending=True).values.contiguous()
    ranks = torch.searchsorted(
        (-descending).contiguous(),
        (-scores).contiguous(),
        right=False,
    ) + 1

    concepts = []
    for token_id, logit, rank in zip(
        token_ids,
        scores.tolist(),
        ranks.tolist(),
        strict=True,
    ):
        metadata = concept_tokens[token_id]
        concepts.append(
            {
                "concept": metadata["concept"],
                "token_text": metadata["token_text"],
                "token_id": token_id,
                "logit": float(logit),
                "rank": int(rank),
            }
        )
    concepts.sort(key=lambda item: (item["rank"], -item["logit"]))
    max_logit = max(concepts, key=lambda item: item["logit"])
    return {
        "best_security_rank": concepts[0]["rank"],
        "max_security_logit": max_logit["logit"],
        "max_security_concept": max_logit["concept"],
        "security_tokens_in_top_10": sum(item["rank"] <= 10 for item in concepts),
        "security_tokens_in_top_50": sum(item["rank"] <= 50 for item in concepts),
        "concepts": concepts,
    }


def build_layer_results(
    logits_by_layer: dict[int, torch.Tensor],
    tokenizer: Any,
    concept_tokens: dict[int, dict[str, str]],
    top_k: int,
) -> dict[str, Any]:
    return {
        str(layer): {
            "top_tokens": top_tokens(logits[0], tokenizer, top_k),
            "security_scores": score_concepts(logits[0], concept_tokens),
        }
        for layer, logits in sorted(logits_by_layer.items())
    }


def aggregate_layers(layer_results: dict[str, Any]) -> dict[str, Any]:
    if not layer_results:
        raise ValueError("Cannot aggregate empty layer results")
    best_rank_item: dict[str, Any] | None = None
    max_logit_item: dict[str, Any] | None = None
    layers_with_top_10: list[int] = []
    layers_with_top_50: list[int] = []

    for layer_text, result in layer_results.items():
        layer = int(layer_text)
        scores = result["security_scores"]
        rank_candidate = {"layer": layer, **scores["concepts"][0]}
        if best_rank_item is None or (
            rank_candidate["rank"], -rank_candidate["logit"]
        ) < (best_rank_item["rank"], -best_rank_item["logit"]):
            best_rank_item = rank_candidate

        current_max = max(scores["concepts"], key=lambda item: item["logit"])
        logit_candidate = {"layer": layer, **current_max}
        if max_logit_item is None or logit_candidate["logit"] > max_logit_item[
            "logit"
        ]:
            max_logit_item = logit_candidate
        if scores["security_tokens_in_top_10"]:
            layers_with_top_10.append(layer)
        if scores["security_tokens_in_top_50"]:
            layers_with_top_50.append(layer)

    assert best_rank_item is not None and max_logit_item is not None
    return {
        "best_security_rank": best_rank_item["rank"],
        "best_security_layer": best_rank_item["layer"],
        "best_security_concept": best_rank_item["concept"],
        "best_security_token": best_rank_item["token_text"],
        "best_security_logit": best_rank_item["logit"],
        "max_security_logit": max_logit_item["logit"],
        "max_logit_layer": max_logit_item["layer"],
        "max_logit_concept": max_logit_item["concept"],
        "layers_with_security_top_10": layers_with_top_10,
        "layers_with_security_top_50": layers_with_top_50,
    }


def extract_sample_readout(
    model: LensModel,
    lens: JacobianLens,
    tokenizer: Any,
    sample: dict[str, Any],
    concept_tokens: dict[int, dict[str, str]],
    *,
    top_k: int = 20,
    max_seq_len: int = 512,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = render_security_prompt(sample, tokenizer)
    jlens_logits, logit_logits, model_logits, input_ids = lens.apply_comparison(
        model,
        prompt,
        layers=lens.source_layers,
        positions=[-1],
        max_seq_len=max_seq_len,
    )
    jlens_layers = build_layer_results(
        jlens_logits, tokenizer, concept_tokens, top_k
    )
    logit_layers = build_layer_results(
        logit_logits, tokenizer, concept_tokens, top_k
    )
    final_vector = model_logits[0]
    return {
        "id": sample["id"],
        "split": sample.get("split"),
        "category": sample["category"],
        "is_injection": bool(sample["is_injection"]),
        "carrier": sample["carrier"],
        "attack_family": sample.get("attack_family"),
        "attack_goal": sample.get("attack_goal"),
        "expected_action": sample["expected_action"],
        "requested_tool": sample.get("requested_tool"),
        "scenario_id": sample.get("scenario_id"),
        "monitor": {
            "layers": lens.source_layers,
            "position": -1,
            "position_description": "Assistant prefix immediately before response",
            "prompt_tokens": int(input_ids.shape[1]),
        },
        "jlens": {
            "layers": jlens_layers,
            "aggregate": aggregate_layers(jlens_layers),
        },
        "logit_lens": {
            "layers": logit_layers,
            "aggregate": aggregate_layers(logit_layers),
        },
        "model_final": {
            "top_tokens": top_tokens(final_vector, tokenizer, top_k),
            "security_scores": score_concepts(final_vector, concept_tokens),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _sample_ids_digest(samples: list[dict[str, Any]]) -> str:
    serialized = json.dumps(
        [sample["id"] for sample in samples],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _item_path(checkpoint_dir: Path, sample_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
        raise ValueError(f"Unsafe sample id for checkpoint path: {sample_id!r}")
    return checkpoint_dir / f"{sample_id}.json"


ProgressCallback = Callable[[int, int, dict[str, Any], bool], None]


def extract_dataset_readouts(
    model: LensModel,
    lens: JacobianLens,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    split: str,
    dataset_path: Path,
    model_path: Path,
    lens_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    top_k: int = 20,
    max_seq_len: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extract a split, atomically checkpointing each sample for restart/resume."""
    if not samples:
        raise ValueError("No samples to extract")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "split": split,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_ids_sha256": _sample_ids_digest(samples),
        "n_samples": len(samples),
        "model_path": str(model_path),
        "lens_path": str(lens_path),
        "lens_n_prompts": lens.n_prompts,
        "lens_d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "top_k": top_k,
        "max_seq_len": max_seq_len,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"Readout checkpoint config mismatch at {manifest_path}. "
                "Use a new run name."
            )
    else:
        atomic_write_json(manifest_path, manifest)

    concept_tokens = build_concept_tokens(tokenizer)
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        item_path = _item_path(checkpoint_dir, sample["id"])
        resumed = item_path.exists()
        if resumed:
            result = json.loads(item_path.read_text(encoding="utf-8"))
            if result.get("id") != sample["id"]:
                raise ValueError(f"Corrupt checkpoint item: {item_path}")
        else:
            result = extract_sample_readout(
                model,
                lens,
                tokenizer,
                sample,
                concept_tokens,
                top_k=top_k,
                max_seq_len=max_seq_len,
            )
            atomic_write_json(item_path, result)
        results.append(result)
        if progress is not None:
            progress(index, len(samples), result, resumed)

    output = {
        "experiment": manifest,
        "concept_token_count": len(concept_tokens),
        "samples": results,
    }
    atomic_write_json(output_path, output)
    return output
