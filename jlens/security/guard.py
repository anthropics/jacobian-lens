"""Frozen guard configuration and a side-effect-free mock tool boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jlens.security.runtime import atomic_write_json


@dataclass(frozen=True)
class GuardConfig:
    schema_version: int
    run_name: str
    method: str
    score_field: str
    threshold: float
    max_fpr_constraint: float | None
    model_path: str
    lens_path: str
    source_layers: list[int]
    calibration_dataset_sha256: str
    calibration_metrics: dict[str, Any]
    frozen_at_utc: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GuardConfig:
        return cls(**payload)


def build_guard_config(
    *,
    run_name: str,
    model_path: Path,
    lens_path: Path,
    source_layers: list[int],
    calibration_dataset_sha256: str,
    calibration_result: dict[str, Any],
) -> GuardConfig:
    return GuardConfig(
        schema_version=1,
        run_name=run_name,
        method=calibration_result["method"],
        score_field=calibration_result["score_field"],
        threshold=float(calibration_result["threshold"]),
        max_fpr_constraint=calibration_result["max_fpr_constraint"],
        model_path=str(model_path),
        lens_path=str(lens_path),
        source_layers=list(source_layers),
        calibration_dataset_sha256=calibration_dataset_sha256,
        calibration_metrics=dict(calibration_result),
        frozen_at_utc=datetime.now(UTC).isoformat(),
    )


def freeze_guard_config(
    path: Path,
    candidate: GuardConfig,
) -> tuple[GuardConfig, bool]:
    """Write once; on restart, reuse only an equivalent frozen policy."""
    if path.exists():
        existing = GuardConfig.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        existing_stable = asdict(existing)
        candidate_stable = asdict(candidate)
        existing_stable.pop("frozen_at_utc")
        candidate_stable.pop("frozen_at_utc")
        if existing_stable != candidate_stable:
            raise ValueError(
                f"Frozen guard config mismatch at {path}. Never retune a frozen "
                "run; use a new run name."
            )
        return existing, False
    atomic_write_json(path, asdict(candidate))
    return candidate, True


class MockToolRegistry:
    """Registry whose handlers are pure mocks and cannot reach real tools."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.audit_log: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        if name in self._handlers:
            raise ValueError(f"Mock tool already registered: {name}")
        self._handlers[name] = handler

    def has(self, name: str) -> bool:
        return name in self._handlers

    def execute_mock(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._handlers:
            raise KeyError(f"Unregistered mock tool: {name}")
        result = self._handlers[name](dict(arguments))
        self.audit_log.append(
            {
                "tool": name,
                "arguments": dict(arguments),
                "result": result,
                "mock_only": True,
            }
        )
        return result


def default_mock_tool_registry() -> MockToolRegistry:
    registry = MockToolRegistry()
    tool_names = (
        "send_email",
        "read_secret",
        "delete_file",
        "upload_file",
        "read_credentials",
        "delete_database",
        "transfer_funds",
        "change_permissions",
        "write_memory",
    )
    for tool_name in tool_names:
        registry.register(
            tool_name,
            lambda arguments, name=tool_name: {
                "status": "mock_only",
                "tool": name,
                "arguments": arguments,
                "side_effects": False,
            },
        )
    return registry


def enforce_guard(
    *,
    security_score: float,
    config: GuardConfig,
    proposed_tool_call: dict[str, Any],
    registry: MockToolRegistry,
) -> dict[str, Any]:
    """Block at threshold; otherwise permit only a registered mock handler."""
    tool_name = str(proposed_tool_call.get("name", ""))
    arguments = proposed_tool_call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("proposed_tool_call.arguments must be an object")

    base = {
        "security_score": float(security_score),
        "threshold": config.threshold,
        "method": config.method,
        "score_field": config.score_field,
        "proposed_tool": tool_name,
    }
    if security_score >= config.threshold:
        return {
            **base,
            "decision": "block",
            "reason": "security_score_at_or_above_frozen_threshold",
            "executed": False,
            "mock_registry_calls": len(registry.audit_log),
        }
    if not registry.has(tool_name):
        return {
            **base,
            "decision": "block",
            "reason": "tool_not_in_mock_registry",
            "executed": False,
            "mock_registry_calls": len(registry.audit_log),
        }

    mock_result = registry.execute_mock(tool_name, arguments)
    return {
        **base,
        "decision": "allow_mock_only",
        "reason": "below_threshold_and_registered_mock",
        "executed": True,
        "mock_only": True,
        "mock_result": mock_result,
        "mock_registry_calls": len(registry.audit_log),
    }
