import argparse
from pathlib import Path

import torch

import jlens
from jlens.security import (
    extract_dataset_readouts,
    load_hf_lens_model,
    validate_security_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract resumable multi-layer security readouts."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/security/smoke_prompts.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/gpu/multilayer_security_readouts.json"),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=512)
    return parser.parse_args()


def progress(index: int, total: int, result: dict, resumed: bool) -> None:
    aggregate = result["jlens"]["aggregate"]
    status = "resume" if resumed else "compute"
    print(
        f"[{index:04d}/{total:04d}] {status:<7} {result['id']:<32} "
        f"rank={aggregate['best_security_rank']:<7} "
        f"layer={aggregate['best_security_layer']}"
    )


def main() -> None:
    args = parse_args()
    samples, report = validate_security_dataset(args.data_path)
    split = args.split or report["split"]
    if any(sample["split"] != split for sample in samples):
        raise ValueError(f"Dataset rows do not all belong to split {split!r}")
    checkpoint_dir = args.checkpoint_dir or (
        args.output_path.parent / f"{args.output_path.stem}_items"
    )

    tokenizer, hf_model, model, info = load_hf_lens_model(
        args.model_path,
        device="cuda",
    )
    lens = jlens.JacobianLens.load(str(args.lens_path))
    if lens.d_model != model.d_model:
        raise ValueError(
            f"Lens d_model={lens.d_model}, model d_model={model.d_model}"
        )
    print(info)
    output = extract_dataset_readouts(
        model,
        lens,
        tokenizer,
        samples,
        split=split,
        dataset_path=args.data_path,
        model_path=args.model_path,
        lens_path=args.lens_path,
        output_path=args.output_path,
        checkpoint_dir=checkpoint_dir,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
        progress=progress,
    )
    print(f"Saved {len(output['samples'])} readouts to {args.output_path.resolve()}")
    del lens, model, hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
