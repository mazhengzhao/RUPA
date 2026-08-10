#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repeated-sampling Harbor agent with uncertainty-based response selection.

For each agent step, this script samples the base LLM multiple times with the
same Harbor prompt, scores each candidate with one uncertainty method, and keeps
the least uncertain candidate as the real trajectory step.

The implementation is intentionally lightweight:
  - CLI mode can run one Harbor step from a prompt file or messages JSON.
  - The RepeatedSamplingConfidenceAgent class can be imported by a Harbor agent
    wrapper and called at each inference step.
  - LiteLLM is used so existing OpenAI-compatible Harbor model settings work.

Example:
    python uncertainty_sampling_harbor_agent.py \\
      --model openai/Qwen3.5-27B \\
      --prompt-file /path/to/agent/episode-0/prompt.txt \\
      --method trajectory_tau \\
      --num-samples 5 \\
      --temperature 0.7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
TRAJECTORY_TAU_PATH = (
    PROJECT_ROOT / "agent-tracer" / "src" / "tau2" / "metrics" / "trajectory_tau.py"
)


def load_trajectory_tau_module():
    spec = importlib.util.spec_from_file_location(
        "agent_confidence_trajectory_tau_agent", TRAJECTORY_TAU_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trajectory_tau.py from {TRAJECTORY_TAU_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_mean(values: list[float]) -> Optional[float]:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return float(np.mean(values))


def entropy_from_logprobs(logprobs: list[float]) -> Optional[float]:
    if not logprobs:
        return None
    return float(-np.mean(np.asarray(logprobs, dtype=np.float64)))


def extract_response_logprobs(response: Any) -> list[float]:
    """Extract token logprobs from LiteLLM/OpenAI-compatible responses."""

    try:
        choice = response.choices[0]
    except Exception:
        return []

    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None and isinstance(choice, dict):
        logprobs_obj = choice.get("logprobs")

    if hasattr(logprobs_obj, "model_dump"):
        logprobs_obj = logprobs_obj.model_dump()
    elif hasattr(logprobs_obj, "dict"):
        logprobs_obj = logprobs_obj.dict()

    if isinstance(logprobs_obj, dict):
        content = logprobs_obj.get("content", [])
    else:
        content = getattr(logprobs_obj, "content", []) if logprobs_obj else []

    logprobs: list[float] = []
    for item in content or []:
        if isinstance(item, dict):
            value = finite_float(item.get("logprob"))
        else:
            value = finite_float(getattr(item, "logprob", None))
        if value is not None:
            logprobs.append(value)
    return logprobs


def response_to_candidate_message(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    message = choice.message

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")

    normalized_tool_calls = []
    for tool_call in tool_calls or []:
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            normalized_tool_calls.append(
                {
                    "tool_call_id": tool_call.get("id"),
                    "function_name": function.get("name") or tool_call.get("name"),
                    "arguments": _parse_json_maybe(function.get("arguments")),
                }
            )
        else:
            function = getattr(tool_call, "function", None)
            normalized_tool_calls.append(
                {
                    "tool_call_id": getattr(tool_call, "id", None),
                    "function_name": getattr(function, "name", None),
                    "arguments": _parse_json_maybe(getattr(function, "arguments", None)),
                }
            )

    return {
        "source": "agent",
        "role": "assistant",
        "message": content or "",
        "tool_calls": normalized_tool_calls,
    }


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value or {}
    try:
        return json.loads(value)
    except Exception:
        return value


@dataclass
class Candidate:
    sample_id: int
    content: str
    uncertainty: Optional[float]
    method: str
    logprobs: list[float]
    metrics: dict[str, Any]
    raw_response: Any

    @property
    def is_scoreable(self) -> bool:
        return self.uncertainty is not None and math.isfinite(self.uncertainty)


@dataclass
class AgentConfig:
    model: str
    method: str = "trajectory_tau"
    num_samples: int = 5
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    top_logprobs: Optional[int] = None


class RepeatedSamplingConfidenceAgent:
    """
    Selects one model step from repeated samples by uncertainty.

    Lower uncertainty is considered better. For trajectory_tau, the candidate is
    appended to the accepted history before scoring, so the score reflects both
    the current generation and its propagated trajectory risk.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_kwargs: Optional[dict[str, Any]] = None,
        trajectory_tau_config: Optional[Any] = None,
    ):
        self.config = config
        self.llm_kwargs = dict(llm_kwargs or {})
        self.accepted_steps: list[dict[str, Any]] = []
        self.trajectory_tau_module = None
        self.trajectory_tau_config = trajectory_tau_config

        if config.method == "trajectory_tau":
            self.trajectory_tau_module = load_trajectory_tau_module()
            cfg_cls = self.trajectory_tau_module.TrajectoryTAUConfig
            if self.trajectory_tau_config is None:
                self.trajectory_tau_config = cfg_cls()
            elif isinstance(self.trajectory_tau_config, dict):
                self.trajectory_tau_config = cfg_cls(**self.trajectory_tau_config)

    def select_next_response(
        self,
        messages: list[dict[str, Any]],
        goal_text: str = "",
        history_steps: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Candidate, list[Candidate]]:
        candidates = self.sample_candidates(messages, goal_text, history_steps)
        selected = self.select_candidate(candidates)
        selected_step = self._candidate_to_step(selected)
        self.accepted_steps.append(selected_step)
        return selected, candidates

    def sample_candidates(
        self,
        messages: list[dict[str, Any]],
        goal_text: str = "",
        history_steps: Optional[list[dict[str, Any]]] = None,
    ) -> list[Candidate]:
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError("litellm is required: pip install litellm") from exc

        candidates = []
        base_kwargs = self._completion_kwargs()
        for sample_id in range(self.config.num_samples):
            kwargs = dict(base_kwargs)
            seed = self._sample_seed(sample_id)
            if seed is not None:
                kwargs["seed"] = seed

            response = completion(
                model=self.config.model,
                messages=messages,
                **kwargs,
            )
            logprobs = extract_response_logprobs(response)
            candidate_step = response_to_candidate_message(response)
            candidate_step["metrics"] = {"logprobs": logprobs}
            uncertainty, metrics = self.score_candidate(
                candidate_step,
                logprobs,
                goal_text,
                history_steps,
            )
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    content=candidate_step["message"],
                    uncertainty=uncertainty,
                    method=self.config.method,
                    logprobs=logprobs,
                    metrics=metrics,
                    raw_response=response,
                )
            )
        return candidates

    def score_candidate(
        self,
        candidate_step: dict[str, Any],
        logprobs: list[float],
        goal_text: str,
        history_steps: Optional[list[dict[str, Any]]],
    ) -> tuple[Optional[float], dict[str, Any]]:
        entropy = entropy_from_logprobs(logprobs)

        if self.config.method == "entropy":
            return entropy, {"entropy": entropy}

        if self.config.method == "tracer":
            return entropy, {"tracer_score": entropy, "entropy": entropy}

        if self.config.method == "saup":
            saup = None if entropy is None else float(math.sqrt(entropy * entropy))
            return saup, {"saup_score": saup, "entropy": entropy}

        if self.config.method == "trajectory_tau":
            if self.trajectory_tau_module is None:
                raise RuntimeError("trajectory_tau module was not loaded")
            steps = list(history_steps if history_steps is not None else self.accepted_steps)
            steps.append(candidate_step)
            result = self.trajectory_tau_module.calculate_trajectory_tau_score(
                steps,
                goal_text=goal_text,
                config=self.trajectory_tau_config,
            )
            score = result.get("tau_score")
            return finite_float(score), result

        raise ValueError(f"Unknown uncertainty method: {self.config.method}")

    def select_candidate(self, candidates: list[Candidate]) -> Candidate:
        if not candidates:
            raise ValueError("No candidates were sampled")
        scoreable = [candidate for candidate in candidates if candidate.is_scoreable]
        if scoreable:
            return min(scoreable, key=lambda item: (item.uncertainty, item.sample_id))
        return candidates[0]

    def _candidate_to_step(self, candidate: Candidate) -> dict[str, Any]:
        step = response_to_candidate_message(candidate.raw_response)
        step["metrics"] = {
            "logprobs": candidate.logprobs,
            "selection_uncertainty": candidate.uncertainty,
            "selection_method": candidate.method,
        }
        return step

    def _completion_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.llm_kwargs)
        kwargs.setdefault("temperature", self.config.temperature)
        kwargs.setdefault("top_p", self.config.top_p)
        kwargs.setdefault("logprobs", True)
        if self.config.top_logprobs is not None:
            kwargs.setdefault("top_logprobs", self.config.top_logprobs)
        if self.config.max_tokens is not None:
            kwargs.setdefault("max_tokens", self.config.max_tokens)
        return kwargs

    def _sample_seed(self, sample_id: int) -> Optional[int]:
        if self.config.seed is None:
            return None
        rng = random.Random(self.config.seed + sample_id)
        return rng.randint(0, 2**31 - 1)


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.messages_json:
        data = json.loads(Path(args.messages_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--messages-json must contain a JSON list of messages")
        return data

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        return [{"role": "user", "content": prompt}]

    if args.prompt:
        return [{"role": "user", "content": args.prompt}]

    raise ValueError("Provide one of --messages-json, --prompt-file, or --prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Harbor agent step with repeated sampling and uncertainty selection."
    )
    parser.add_argument("--model", required=True, help="LiteLLM model name, e.g. openai/Qwen3.5-27B")
    parser.add_argument(
        "--method",
        choices=["entropy", "tracer", "saup", "trajectory_tau"],
        default="trajectory_tau",
        help="Uncertainty method used to rank repeated samples.",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-logprobs", type=int, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--messages-json", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--goal-text", type=str, default="")
    parser.add_argument(
        "--history-trajectory",
        type=Path,
        default=None,
        help="Optional Harbor trajectory.json; existing steps are used for trajectory_tau scoring.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the selected response and candidate diagnostics.",
    )
    return parser.parse_args()


def load_history(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("steps", []))


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    config = AgentConfig(
        model=args.model,
        method=args.method,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_logprobs=args.top_logprobs,
    )
    agent = RepeatedSamplingConfidenceAgent(config)
    messages = load_messages(args)
    history_steps = load_history(args.history_trajectory)

    selected, candidates = agent.select_next_response(
        messages=messages,
        goal_text=args.goal_text,
        history_steps=history_steps,
    )

    payload = {
        "selected_sample_id": selected.sample_id,
        "selected_uncertainty": selected.uncertainty,
        "method": selected.method,
        "response": selected.content,
        "candidates": [
            {
                "sample_id": candidate.sample_id,
                "uncertainty": candidate.uncertainty,
                "n_logprobs": len(candidate.logprobs),
                "metrics": candidate.metrics,
                "response": candidate.content,
            }
            for candidate in candidates
        ],
        "config": asdict(config),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(selected.content)
    print(
        json.dumps(
            {
                "selected_sample_id": selected.sample_id,
                "selected_uncertainty": selected.uncertainty,
                "method": selected.method,
                "num_candidates": len(candidates),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
