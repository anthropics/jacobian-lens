from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from jlens.fitting import fit
from jlens.security import (
    build_guard_config,
    calibrate_all_methods,
    confusion_matrices_svg,
    default_mock_tool_registry,
    enforce_guard,
    evaluate_all_methods,
    extract_dataset_readouts,
    freeze_guard_config,
    metric_bar_chart_svg,
    score_concepts,
    validate_security_dataset,
)

from .tiny import TinyDecoder


class _SecurityTokenizer:
    bos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
    ) -> SimpleNamespace:
        ids = [0] + [1 + byte % 30 for byte in text.encode()][: max_length - 1]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [1 + sum(text.encode()) % 30]

    def decode(self, ids, **_kwargs) -> str:
        return f"tok_{int(ids[0])}"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return "\n".join(message["content"] for message in messages) + "\nassistant:"


def _sample(sample_id: str, category: str, split: str = "smoke") -> dict:
    injection = category == "injection"
    return {
        "id": sample_id,
        "split": split,
        "category": category,
        "is_injection": injection,
        "carrier": "document",
        "user_task": "Summarize this document.",
        "external_content": "A sufficiently long external document for testing.",
        "attack_family": "instruction_override" if injection else None,
        "attack_goal": "data_exfiltration" if injection else None,
        "requested_tool": "send_email" if injection else None,
        "expected_action": "block" if injection else "allow",
        "language": "en",
        "source": "test",
        "template_id": f"template_{sample_id}",
        "scenario_id": f"scenario_{sample_id}",
    }


def test_validate_security_dataset(tmp_path: Path):
    path = tmp_path / "smoke.jsonl"
    samples = [
        _sample("clean", "clean"),
        _sample("hard", "hard_negative"),
        _sample("attack", "injection"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in samples), encoding="utf-8")
    loaded, report = validate_security_dataset(path, expected_split="smoke")
    assert loaded == samples
    assert report["n_samples"] == 3
    assert report["n_injection"] == 1


def test_score_concepts_matches_brute_force():
    logits = torch.tensor([0.5, -1.0, 3.0, 2.0, 2.0])
    tokens = {
        0: {"concept": "zero", "token_text": "zero"},
        3: {"concept": "three", "token_text": "three"},
    }
    scored = score_concepts(logits, tokens)
    by_id = {item["token_id"]: item for item in scored["concepts"]}
    assert by_id[0]["rank"] == int((logits > logits[0]).sum()) + 1
    assert by_id[3]["rank"] == int((logits > logits[3]).sum()) + 1


def test_readout_checkpoint_resume(tmp_path: Path):
    model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32)
    model.tokenizer = _SecurityTokenizer()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lens = fit(
        model,
        ["generic corpus text " * 8],
        source_layers=[1, 2],
        dim_batch=4,
        max_seq_len=64,
        skip_first=2,
    )
    samples = [
        _sample("clean", "clean"),
        _sample("hard", "hard_negative"),
        _sample("attack", "injection"),
    ]
    dataset_path = tmp_path / "smoke.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in samples), encoding="utf-8"
    )
    resumed_flags: list[bool] = []
    output = extract_dataset_readouts(
        model,
        lens,
        model.tokenizer,
        samples,
        split="smoke",
        dataset_path=dataset_path,
        model_path=tmp_path / "model",
        lens_path=tmp_path / "lens.pt",
        output_path=tmp_path / "readouts.json",
        checkpoint_dir=tmp_path / "items",
        top_k=5,
        progress=lambda _i, _n, _r, resumed: resumed_flags.append(resumed),
    )
    assert len(output["samples"]) == 3
    assert resumed_flags == [False, False, False]

    resumed_flags.clear()
    second = extract_dataset_readouts(
        model,
        lens,
        model.tokenizer,
        samples,
        split="smoke",
        dataset_path=dataset_path,
        model_path=tmp_path / "model",
        lens_path=tmp_path / "lens.pt",
        output_path=tmp_path / "readouts.json",
        checkpoint_dir=tmp_path / "items",
        top_k=5,
        progress=lambda _i, _n, _r, resumed: resumed_flags.append(resumed),
    )
    assert second == output
    assert resumed_flags == [True, True, True]


def _metric_rows() -> list[dict]:
    rows = []
    for method, offset in (("jlens", 0.4), ("logit_lens", 0.2), ("model_final", 0.1)):
        for index, (label, score) in enumerate(
            ((0, 0.1), (0, 0.2), (0, 0.3), (1, 0.7), (1, 0.8), (1, 0.9))
        ):
            rows.append(
                {
                    "id": f"{method}_{index}",
                    "split": "calibration",
                    "category": "injection" if label else "clean",
                    "is_injection": label,
                    "carrier": "document",
                    "attack_family": None,
                    "method": method,
                    "best_rank": 1,
                    "max_logit": score + offset,
                    "rank_score": 0.0,
                    "elapsed_seconds": 0.1,
                }
            )
    return rows


def test_calibration_metrics_and_mock_guard(tmp_path: Path):
    calibration = calibrate_all_methods(_metric_rows(), max_fpr=0.0)
    metrics = evaluate_all_methods(_metric_rows(), calibration)
    assert all(row["f1"] == 1.0 for row in metrics)
    assert "<svg" in metric_bar_chart_svg(metrics, title="metrics")
    assert "<svg" in confusion_matrices_svg(metrics, title="confusion")

    candidate = build_guard_config(
        run_name="test",
        model_path=tmp_path / "model",
        lens_path=tmp_path / "lens.pt",
        source_layers=[1, 2],
        calibration_dataset_sha256="abc",
        calibration_result=calibration["jlens"],
    )
    path = tmp_path / "guard.json"
    config, created = freeze_guard_config(path, candidate)
    assert created
    assert freeze_guard_config(path, candidate)[1] is False

    registry = default_mock_tool_registry()
    blocked = enforce_guard(
        security_score=config.threshold,
        config=config,
        proposed_tool_call={"name": "delete_file", "arguments": {"path": "x"}},
        registry=registry,
    )
    assert blocked["decision"] == "block"
    assert not blocked["executed"]
    assert registry.audit_log == []
