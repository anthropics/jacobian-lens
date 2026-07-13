import argparse
import os
from pathlib import Path

import torch

from jlens.security import (
    fit_or_load_lens,
    load_hf_lens_model,
    load_or_cache_generic_prompts,
)

DEFAULT_DIM_BATCH = 52
DEFAULT_MIN_FREE_VRAM_GIB = 11.0


def parse_layers(value: str) -> list[int]:
    return sorted(
        {
            int(layer.strip())
            for layer in value.split(",")
            if layer.strip()
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit or resume a multi-layer Jacobian lens on GPU."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lenses"),
    )
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--source-layers", default="18,24,30,34")
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--generic-corpus", choices=("wikitext", "builtin"), default="wikitext")
    parser.add_argument("--run-name")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        default=DEFAULT_MIN_FREE_VRAM_GIB,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_prompts <= 0 or args.dim_batch <= 0:
        raise ValueError("n_prompts and dim_batch must be > 0")
    if args.skip_first < 0 or args.max_seq_len <= args.skip_first + 1:
        raise ValueError("max_seq_len must be greater than skip_first + 1")
    if args.checkpoint_every <= 0 or args.min_free_vram_gib < 0:
        raise ValueError("checkpoint_every must be > 0 and min VRAM must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    source_layers = parse_layers(args.source_layers)
    layer_slug = "-".join(map(str, source_layers))
    run_name = args.run_name or (
        f"qwen3b_gpu_n{args.n_prompts}_seq{args.max_seq_len}"
        f"_db{args.dim_batch}_layers{layer_slug}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lens_path = args.output_dir / f"{run_name}_lens.pt"
    checkpoint_path = args.output_dir / f"{run_name}_checkpoint.pt"
    manifest_path = args.output_dir / f"{run_name}_manifest.json"
    prompt_cache = args.output_dir / f"{run_name}_generic_prompts.json"

    tokenizer, hf_model, model, model_info = load_hf_lens_model(
        args.model_path,
        device="cuda",
        min_free_vram_gib=args.min_free_vram_gib,
    )
    del tokenizer
    print("=== Model ===")
    for key, value in model_info.items():
        print(f"{key:<28}: {value}")

    target_layer = args.target_layer + model.n_layers if args.target_layer < 0 else args.target_layer
    if not source_layers or source_layers[0] < 0 or source_layers[-1] >= target_layer:
        raise ValueError(f"source_layers must be within [0, {target_layer})")

    prompts = load_or_cache_generic_prompts(
        prompt_cache,
        n_prompts=args.n_prompts,
        source=args.generic_corpus,
    )
    lens, fit_info = fit_or_load_lens(
        model,
        prompts,
        model_path=args.model_path,
        lens_path=lens_path,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_every=args.checkpoint_every,
        force_refit=args.no_resume,
    )
    print("\n=== Lens ===")
    print(lens)
    for key, value in fit_info.items():
        print(f"{key:<28}: {value}")
    del model, hf_model


if __name__ == "__main__":
    main()
