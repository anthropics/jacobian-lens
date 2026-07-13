from pathlib import Path

import jlens
from jlens.security import (
    extract_dataset_readouts,
    load_hf_lens_model,
    validate_security_dataset,
)

MODEL_PATH = Path("models/Qwen2.5-3B-Instruct")
LENS_PATH = Path("artifacts/lenses/qwen3b_cpu_smoke_lens.pt")
DATA_PATH = Path("data/security/smoke_prompts.jsonl")
OUTPUT_PATH = Path("results/smoke/security_readouts.json")
CHECKPOINT_DIR = Path("results/smoke/security_readout_items")


def progress(index: int, total: int, result: dict, resumed: bool) -> None:
    status = "resume" if resumed else "compute"
    aggregate = result["jlens"]["aggregate"]
    print(
        f"[{index:02d}/{total:02d}] {status:<7} {result['id']:<30} "
        f"rank={aggregate['best_security_rank']}"
    )


def main() -> None:
    samples, _ = validate_security_dataset(DATA_PATH, expected_split="smoke")
    tokenizer, hf_model, model, info = load_hf_lens_model(
        MODEL_PATH,
        device="cpu",
    )
    lens = jlens.JacobianLens.load(str(LENS_PATH))
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
        split="smoke",
        dataset_path=DATA_PATH,
        model_path=MODEL_PATH,
        lens_path=LENS_PATH,
        output_path=OUTPUT_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        top_k=20,
        progress=progress,
    )
    print(f"Saved {len(output['samples'])} readouts to {OUTPUT_PATH.resolve()}")
    del model, hf_model


if __name__ == "__main__":
    main()
