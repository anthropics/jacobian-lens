import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data/security/smoke_prompts.jsonl")

REQUIRED_FIELDS = {
    "id",
    "category",
    "is_injection",
    "carrier",
    "user_task",
    "external_content",
    "attack_family",
    "requested_tool",
    "expected_action",
}

VALID_CATEGORIES = {
    "clean",
    "hard_negative",
    "injection",
}


def main() -> None:
    samples = []
    seen_ids = set()

    with DATA_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            sample = json.loads(line)

            missing = REQUIRED_FIELDS - sample.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing fields {sorted(missing)}"
                )

            if sample["id"] in seen_ids:
                raise ValueError(
                    f"Line {line_number}: duplicate ID {sample['id']}"
                )

            if sample["category"] not in VALID_CATEGORIES:
                raise ValueError(
                    f"Line {line_number}: invalid category "
                    f"{sample['category']}"
                )

            expected_label = sample["category"] == "injection"

            if sample["is_injection"] != expected_label:
                raise ValueError(
                    f"Line {line_number}: inconsistent injection label"
                )

            seen_ids.add(sample["id"])
            samples.append(sample)

    counts = Counter(sample["category"] for sample in samples)

    print(f"Dataset: {DATA_PATH}")
    print(f"Total samples: {len(samples)}")

    for category in sorted(VALID_CATEGORIES):
        print(f"{category}: {counts[category]}")

    assert len(samples) == 9
    assert counts["clean"] == 3
    assert counts["hard_negative"] == 3
    assert counts["injection"] == 3

    print("Dataset validation completed successfully.")


if __name__ == "__main__":
    main()