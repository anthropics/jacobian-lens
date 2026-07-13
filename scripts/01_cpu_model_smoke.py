import os
import time
from pathlib import Path

import jlens
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path("models/Qwen2.5-0.5B-Instruct")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {MODEL_PATH.resolve()}"
        )

    # Không để PyTorch chiếm toàn bộ CPU threads.
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(8, cpu_count))

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CPU threads     : {torch.get_num_threads()}")
    print(f"Model path      : {MODEL_PATH.resolve()}")

    print("\n[1/4] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("[2/4] Loading model on CPU...")

    load_started = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    load_seconds = time.perf_counter() - load_started

    print(f"Model loaded in {load_seconds:.2f} seconds")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": (
                "Answer briefly: What is prompt injection "
                "in an AI system?"
            ),
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    print("[3/4] Generating response on CPU...")

    generation_started = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = time.perf_counter() - generation_started

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\nModel response:")
    print(response)

    print(f"\nGeneration time: {generation_seconds:.2f} seconds")
    print(f"Generated tokens: {generated_ids.shape[0]}")

    print("\n[4/4] Wrapping model with jlens.from_hf()...")

    lens_model = jlens.from_hf(
        model,
        tokenizer,
        compile=False,
    )

    print("J-lens wrapper created successfully:")
    print(lens_model)

    print("\nCPU smoke test completed successfully.")


if __name__ == "__main__":
    main()