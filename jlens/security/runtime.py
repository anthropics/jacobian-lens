"""Run layout, model loading and resumable generic-corpus lens fitting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.examples import load_wikitext_prompts

GIB = 1024**3
BUILTIN_GENERIC_PROMPTS = [
    (
        "Scientific measurement depends on clear definitions, calibrated "
        "instruments, repeated observation, and careful separation of evidence "
        "from interpretation. Researchers document assumptions, compare alternative "
        "explanations, and preserve enough detail for independent replication. "
        "A useful report describes the question, method, observations, uncertainty, "
        "and limitations before drawing a conclusion."
    ),
    (
        "Cities coordinate transportation, water, energy, housing, parks, and public "
        "services across many time horizons. Planning teams combine historical data "
        "with population forecasts, maintenance schedules, accessibility needs, and "
        "community feedback. Decisions are reviewed periodically because conditions "
        "change and because a robust plan should remain useful under several plausible "
        "future scenarios."
    ),
    (
        "A software release moves through design, implementation, review, automated "
        "testing, staged deployment, observation, and retrospective analysis. Teams "
        "record expected behavior and failure modes, keep changes small enough to "
        "inspect, and use reversible deployment steps. Clear ownership and versioned "
        "artifacts make it easier to diagnose regressions without relying on memory."
    ),
    (
        "Libraries preserve books, recordings, maps, photographs, and digital records "
        "so that people can discover and interpret them over time. Cataloguing adds "
        "structured descriptions, preservation protects fragile material, and access "
        "policies balance availability with legal and ethical constraints. Long-term "
        "stewardship requires routine audits and migration to supported formats."
    ),
    (
        "Ecological systems contain interacting populations, resources, climates, and "
        "landscapes. A local observation may reflect seasonal variation, migration, "
        "measurement error, or a lasting environmental change. Field studies therefore "
        "use repeated sampling, comparison sites, and explicit uncertainty rather than "
        "treating a single measurement as a complete description of the system."
    ),
    (
        "Manufacturing quality is supported by stable processes, traceable materials, "
        "well-defined tolerances, and feedback from inspection. When a defect appears, "
        "engineers examine the process history and test candidate causes instead of "
        "changing many variables at once. Corrective actions are monitored to confirm "
        "that they improve reliability without creating a new failure mode."
    ),
    (
        "Education involves practice, feedback, retrieval, explanation, and transfer "
        "to unfamiliar situations. A learner may recognize an example without being "
        "able to apply the underlying idea, so assessments should sample more than one "
        "form of understanding. Instructors revise pacing and examples using evidence "
        "from student work while keeping learning objectives stable and visible."
    ),
    (
        "Historical analysis compares primary records, later interpretations, material "
        "evidence, and the context in which each source was produced. Sources may be "
        "incomplete or shaped by the interests of their authors. A careful account "
        "therefore distinguishes documented events from inference, identifies conflicts "
        "between records, and explains why one interpretation is better supported."
    ),
]


def bytes_to_gib(value: int) -> float:
    return value / GIB


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_name: str
    run_dir: Path
    config_path: Path
    lens_dir: Path
    lens_path: Path
    fit_checkpoint_path: Path
    fit_manifest_path: Path
    generic_prompts_path: Path
    readouts_dir: Path
    metrics_dir: Path
    guard_dir: Path
    guard_config_path: Path

    @classmethod
    def create(cls, root: Path, run_name: str) -> RunPaths:
        if not run_name or any(character in run_name for character in "/\\"):
            raise ValueError("run_name must be a non-empty path-safe name")
        run_dir = root / "artifacts" / "runs" / run_name
        lens_dir = run_dir / "lens"
        readouts_dir = run_dir / "readouts"
        metrics_dir = run_dir / "metrics"
        guard_dir = run_dir / "guard"
        for directory in (run_dir, lens_dir, readouts_dir, metrics_dir, guard_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            run_name=run_name,
            run_dir=run_dir,
            config_path=run_dir / "run_config.json",
            lens_dir=lens_dir,
            lens_path=lens_dir / "lens.pt",
            fit_checkpoint_path=lens_dir / "fit_checkpoint.pt",
            fit_manifest_path=lens_dir / "fit_manifest.json",
            generic_prompts_path=lens_dir / "generic_prompts.json",
            readouts_dir=readouts_dir,
            metrics_dir=metrics_dir,
            guard_dir=guard_dir,
            guard_config_path=guard_dir / "guard_config.json",
        )


def ensure_run_config(path: Path, config: dict[str, Any]) -> None:
    """Write a run config once and reject accidental cross-run reuse."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                f"Run config mismatch at {path}. Use a new run name or restore "
                "the original configuration."
            )
        return
    atomic_write_json(path, config)


def load_or_cache_generic_prompts(
    cache_path: Path,
    *,
    n_prompts: int,
    source: str = "wikitext",
) -> list[str]:
    """Load a stable generic fit corpus, caching the exact prompt texts."""
    if n_prompts <= 0:
        raise ValueError("n_prompts must be > 0")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("source") != source:
            raise ValueError(
                f"Generic corpus source mismatch at {cache_path}: "
                f"{cached.get('source')!r} != {source!r}"
            )
        prompts = cached.get("prompts", [])
        if len(prompts) < n_prompts:
            raise ValueError(
                f"Generic corpus cache has {len(prompts)} prompts, needs {n_prompts}. "
                "Use a new run name to change corpus size."
            )
        return prompts[:n_prompts]

    if source == "wikitext":
        prompts = load_wikitext_prompts(n_prompts=n_prompts)
    elif source == "builtin":
        if n_prompts > len(BUILTIN_GENERIC_PROMPTS):
            raise ValueError(
                f"builtin corpus supports at most {len(BUILTIN_GENERIC_PROMPTS)} "
                "prompts"
            )
        prompts = BUILTIN_GENERIC_PROMPTS[:n_prompts]
    else:
        raise ValueError(f"Unknown generic corpus source: {source!r}")

    atomic_write_json(
        cache_path,
        {"source": source, "n_prompts": len(prompts), "prompts": prompts},
    )
    return prompts


def load_hf_lens_model(
    model_path: Path,
    *,
    device: str = "cuda",
    min_free_vram_gib: float = 0.0,
) -> tuple[Any, Any, jlens.HFLensModel, dict[str, Any]]:
    """Load a local HF causal LM and wrap it for Jacobian-lens operations."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    target = torch.device(device)
    dtype = (
        torch.bfloat16
        if target.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if target.type == "cuda"
        else torch.float32
    )
    free_vram = total_vram = None
    if target.type == "cuda":
        free_vram, total_vram = torch.cuda.mem_get_info(target)
        if bytes_to_gib(free_vram) < min_free_vram_gib:
            raise RuntimeError(
                f"Only {bytes_to_gib(free_vram):.2f} GiB VRAM is free; "
                f"{min_free_vram_gib:.2f} GiB required. Shut down other GPU jobs."
            )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        ).to(target)
    except torch.OutOfMemoryError as exc:
        if target.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU ran out of memory while loading the model. Shut down other "
            "GPU processes or choose a smaller model."
        ) from exc
    hf_model.eval()
    model = jlens.from_hf(hf_model, tokenizer, compile=False)
    info = {
        "model_path": str(model_path),
        "device": str(target),
        "dtype": str(dtype),
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "load_seconds": time.perf_counter() - started,
    }
    if target.type == "cuda":
        info.update(
            {
                "gpu": torch.cuda.get_device_name(target),
                "total_vram_gib": bytes_to_gib(total_vram),
                "free_vram_before_load_gib": bytes_to_gib(free_vram),
                "model_allocated_gib": bytes_to_gib(
                    torch.cuda.memory_allocated(target)
                ),
                "model_reserved_gib": bytes_to_gib(
                    torch.cuda.memory_reserved(target)
                ),
            }
        )
    return tokenizer, hf_model, model, info


def fit_or_load_lens(
    model: jlens.HFLensModel,
    prompts: list[str],
    *,
    model_path: Path,
    lens_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_every: int = 1,
    force_refit: bool = False,
) -> tuple[jlens.JacobianLens, dict[str, Any]]:
    """Load a completed lens or fit/resume it from the run checkpoint."""
    resolved_target = target_layer + model.n_layers if target_layer < 0 else target_layer
    config = {
        "model_path": str(model_path),
        "n_prompts": len(prompts),
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_layers": source_layers,
        "target_layer": resolved_target,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "checkpoint_every": checkpoint_every,
        "d_model": model.d_model,
    }

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                f"Lens fit config mismatch at {manifest_path}. Use a new run name."
            )
    else:
        atomic_write_json(manifest_path, config)

    if lens_path.exists() and not force_refit:
        lens = jlens.JacobianLens.load(str(lens_path))
        if lens.d_model != model.d_model or lens.source_layers != source_layers:
            raise ValueError("Existing lens is incompatible with the current model")
        return lens, {"status": "loaded", "lens_path": str(lens_path), **config}

    lens_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and model.input_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(model.input_device)
    started = time.perf_counter()
    try:
        lens = jlens.fit(
            model,
            prompts=prompts,
            source_layers=source_layers,
            target_layer=resolved_target,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=checkpoint_every,
            resume=not force_refit,
        )
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        suggestion = max(1, dim_batch // 2)
        raise RuntimeError(
            f"GPU OOM with dim_batch={dim_batch}; retry in a new run with "
            f"dim_batch={suggestion}."
        ) from exc
    lens.save(str(lens_path))

    result = {
        "status": "fitted",
        "lens_path": str(lens_path),
        "fit_seconds": time.perf_counter() - started,
        **config,
    }
    if torch.cuda.is_available() and model.input_device.type == "cuda":
        total = torch.cuda.get_device_properties(model.input_device).total_memory
        peak_allocated = torch.cuda.max_memory_allocated(model.input_device)
        peak_reserved = torch.cuda.max_memory_reserved(model.input_device)
        result.update(
            {
                "peak_allocated_gib": bytes_to_gib(peak_allocated),
                "peak_reserved_gib": bytes_to_gib(peak_reserved),
                "vram_utilization": peak_reserved / total,
                "reserved_headroom_gib": bytes_to_gib(total - peak_reserved),
            }
        )
    return lens, result
