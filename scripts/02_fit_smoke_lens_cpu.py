import logging
import os
import time
from pathlib import Path

import jlens
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path("models/Qwen2.5-3B-Instruct")

ARTIFACT_DIR = Path("artifacts/lenses")
CHECKPOINT_PATH = ARTIFACT_DIR / "qwen3b_cpu_smoke_fit_checkpoint.pt"
LENS_PATH = ARTIFACT_DIR / "qwen3b_cpu_smoke_lens.pt"


FIT_PROMPTS = [
    (
        "Artificial intelligence systems process natural language by "
        "transforming input tokens through multiple neural network layers. "
        "Intermediate representations encode different types of information "
        "before the model predicts the next token."
    )
]


def decode_top_tokens(
    logits: torch.Tensor,
    tokenizer,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    values, indices = logits.topk(top_k)

    results = []

    for token_id, value in zip(
        indices.tolist(),
        values.tolist(),
        strict=True,
    ):
        token = tokenizer.decode([token_id])
        results.append((repr(token), value))

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH.resolve()}"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(8, cpu_count))

    print("=== Environment ===")
    print(f"Torch version : {torch.__version__}")
    print(f"CUDA          : {torch.cuda.is_available()}")
    print(f"CPU threads   : {torch.get_num_threads()}")
    print(f"Model path    : {MODEL_PATH.resolve()}")

    print("\n[1/5] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("[2/5] Loading Qwen 3B in FP32 on CPU...")

    load_started = time.perf_counter()

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        local_files_only=True,
    )

    hf_model.to("cpu")
    hf_model.eval()

    load_seconds = time.perf_counter() - load_started

    print(f"Model loaded in {load_seconds:.2f} seconds")

    print("[3/5] Wrapping Qwen with jlens.from_hf()...")

    model = jlens.from_hf(
        hf_model,
        tokenizer,
        compile=False,
    )

    print(model)
    print(f"Number of layers : {model.n_layers}")
    print(f"Residual width   : {model.d_model}")

    source_layer = model.n_layers - 2
    target_layer = model.n_layers - 1

    print("\n=== Smoke fitting configuration ===")
    print(f"Source layer : {source_layer}")
    print(f"Target layer : {target_layer}")
    print("Prompts      : 1")
    print("Max seq len  : 24")
    print("Skip first   : 2")
    print("Dim batch    : 32")
    print(
        "Backward passes per prompt: "
        f"{(model.d_model + 32 - 1) // 32}"
    )

    print("\n[4/5] Fitting smoke-test Jacobian lens...")

    fit_started = time.perf_counter()

    lens = jlens.fit(
        model,
        prompts=FIT_PROMPTS,
        source_layers=[source_layer],
        target_layer=target_layer,
        dim_batch=32,
        max_seq_len=24,
        skip_first=2,
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_every=1,
        resume=True,
    )

    fit_seconds = time.perf_counter() - fit_started

    print(f"\nFitting completed in {fit_seconds:.2f} seconds")
    print(lens)

    lens.save(str(LENS_PATH))

    print(f"Lens saved to: {LENS_PATH.resolve()}")
    print(
        f"Lens size: {LENS_PATH.stat().st_size / 1024**2:.2f} MiB"
    )

    print("\n[5/5] Reloading and applying the fitted lens...")

    reloaded_lens = jlens.JacobianLens.load(str(LENS_PATH))

    print(f"Reloaded: {reloaded_lens}")

    evaluation_prompt = "The largest planet in the solar system is"

    jlens_logits, model_logits, input_ids = reloaded_lens.apply(
        model,
        evaluation_prompt,
        layers=[source_layer],
        positions=[-1],
        use_jacobian=True,
    )

    logit_lens_logits, _, _ = reloaded_lens.apply(
        model,
        evaluation_prompt,
        layers=[source_layer],
        positions=[-1],
        use_jacobian=False,
    )

    print("\n=== Evaluation prompt ===")
    print(evaluation_prompt)

    print("\nInput tokens:")
    print(tokenizer.convert_ids_to_tokens(input_ids[0]))

    print(f"\nJ-lens top tokens at layer {source_layer}:")

    for token, score in decode_top_tokens(
        jlens_logits[source_layer][0],
        tokenizer,
    ):
        print(f"{token:>20}  {score:>10.4f}")

    print(f"\nLogit-lens top tokens at layer {source_layer}:")

    for token, score in decode_top_tokens(
        logit_lens_logits[source_layer][0],
        tokenizer,
    ):
        print(f"{token:>20}  {score:>10.4f}")

    print("\nModel final top tokens:")

    for token, score in decode_top_tokens(
        model_logits[0],
        tokenizer,
    ):
        print(f"{token:>20}  {score:>10.4f}")

    print("\nSmoke-test J-lens pipeline completed successfully.")


if __name__ == "__main__":
    main()