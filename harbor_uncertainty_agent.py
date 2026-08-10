#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harbor-compatible uncertainty sampling agent.

Use with Harbor's custom agent import path:

    harbor run \
      --dataset terminal-bench/terminal-bench-2 \
      --agent-import-path harbor_uncertainty_agent:UncertaintySamplingTerminus2 \
      --model Qwen3.5-27B \
      --agent-kwarg num_samples=5 \
      --agent-kwarg uncertainty_method=trajectory_tau \
      --agent-kwarg temperature=0.7 \
      --force-build \
      --yes

This class subclasses Harbor's built-in Terminus2 agent. It keeps Harbor's
standard terminal loop, trajectory format, logging, and evaluation behavior, but
replaces each LLM call with repeated sampling. The selected response is the
candidate with the lowest uncertainty score.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat


PROJECT_ROOT = Path(__file__).resolve().parent
TRAJECTORY_TAU_PATH = (
    PROJECT_ROOT / "agent-tracer" / "src" / "tau2" / "metrics" / "trajectory_tau.py"
)


def _load_trajectory_tau_module():
    spec = importlib.util.spec_from_file_location(
        "agent_confidence_harbor_trajectory_tau", TRAJECTORY_TAU_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trajectory_tau.py from {TRAJECTORY_TAU_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _entropy_from_logprobs(logprobs: list[float] | None) -> Optional[float]:
    if not logprobs:
        return None
    cleaned = [_finite_float(x) for x in logprobs]
    values = [x for x in cleaned if x is not None]
    if not values:
        return None
    return float(-np.mean(np.asarray(values, dtype=np.float64)))


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return None


@dataclass
class ScoredResponse:
    sample_id: int
    response: LLMResponse
    uncertainty: Optional[float]
    metrics: dict[str, Any]
    elapsed_ms: float

    @property
    def scoreable(self) -> bool:
        return self.uncertainty is not None and math.isfinite(self.uncertainty)


class UncertaintySamplingTerminus2(Terminus2):
    """
    Terminus2 with repeated per-step sampling and uncertainty-based selection.

    Parameters are passed through Harbor's --agent-kwarg:
      - num_samples: number of candidate generations per step.
      - uncertainty_method: entropy | tracer | saup | trajectory_tau.
      - selection_debug: whether to write candidate diagnostics into episode debug.json.
      - max_turns: hard cap on Terminus2 loop iterations.
      - max_consecutive_repeated_actions: abort after repeated identical actions.

    Lower score is treated as lower uncertainty and therefore selected.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        num_samples: int = 5,
        uncertainty_method: Literal[
            "entropy", "tracer", "saup", "trajectory_tau"
        ] = "trajectory_tau",
        selection_debug: bool = True,
        trajectory_tau_config: dict[str, Any] | None = None,
        max_turns: int | None = 80,
        max_consecutive_repeated_actions: int | None = 8,
        suppress_max_turns_warning: bool = True,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            suppress_max_turns_warning=suppress_max_turns_warning,
            **kwargs,
        )
        self._num_samples = int(num_samples)
        if self._num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self._uncertainty_method = uncertainty_method
        self._selection_debug = bool(selection_debug)
        self._max_consecutive_repeated_actions = (
            int(max_consecutive_repeated_actions)
            if max_consecutive_repeated_actions is not None
            else None
        )
        self._last_action_signature: str | None = None
        self._repeated_action_count = 0
        self._trajectory_tau_module = None
        self._trajectory_tau_config = None
        if uncertainty_method == "trajectory_tau":
            self._trajectory_tau_module = _load_trajectory_tau_module()
            cfg_cls = self._trajectory_tau_module.TrajectoryTAUConfig
            self._trajectory_tau_config = cfg_cls(**(trajectory_tau_config or {}))

    @staticmethod
    def name() -> str:
        return "uncertainty-sampling-terminus-2"

    def version(self) -> str | None:
        return "0.1.0"

    def _reset_per_run_state(self) -> None:
        super()._reset_per_run_state()
        self._last_action_signature = None
        self._repeated_action_count = 0

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session=None,
    ):
        result = await super()._handle_llm_interaction(
            chat=chat,
            prompt=prompt,
            logging_paths=logging_paths,
            original_instruction=original_instruction,
            session=session,
        )
        commands, is_task_complete = result[0], result[1]
        self._check_repeated_actions(commands, is_task_complete)
        return result

    def _check_repeated_actions(self, commands: list[Any], is_task_complete: bool) -> None:
        if self._max_consecutive_repeated_actions is None:
            return
        if self._max_consecutive_repeated_actions <= 0:
            return
        if is_task_complete:
            self._last_action_signature = None
            self._repeated_action_count = 0
            return

        signature = self._action_signature(commands)
        if signature == self._last_action_signature:
            self._repeated_action_count += 1
        else:
            self._last_action_signature = signature
            self._repeated_action_count = 1

        if self._repeated_action_count >= self._max_consecutive_repeated_actions:
            raise RuntimeError(
                "Aborting trial after repeated identical agent actions "
                f"({self._repeated_action_count} consecutive repeats): {signature}"
            )

    @staticmethod
    def _action_signature(commands: list[Any]) -> str:
        if not commands:
            return "<no_commands>"
        parts = []
        for command in commands:
            keystrokes = getattr(command, "keystrokes", "")
            normalized = " ".join(str(keystrokes).strip().split())
            parts.append(normalized[:500])
        return "\n---\n".join(parts)

    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session=None,
    ) -> LLMResponse:
        logging_path, prompt_path, response_path = logging_paths

        if prompt_path is not None:
            prompt_path.write_text(prompt)

        try:
            scored: list[ScoredResponse] = []
            for sample_id in range(self._num_samples):
                start_time = time.time()
                response = await self._llm.call(
                    prompt=prompt,
                    message_history=copy.deepcopy(chat.messages),
                    logging_path=None,
                    previous_response_id=getattr(chat, "_last_response_id", None),
                    **self._llm_call_kwargs,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                self._api_request_times.append(elapsed_ms)
                uncertainty, metrics = self._score_response(
                    response=response,
                    prompt=prompt,
                    original_instruction=original_instruction,
                )
                scored.append(
                    ScoredResponse(
                        sample_id=sample_id,
                        response=response,
                        uncertainty=uncertainty,
                        metrics=metrics,
                        elapsed_ms=elapsed_ms,
                    )
                )
        except ContextLengthExceededError:
            self.logger.debug(
                "Context length exceeded during uncertainty sampling; falling back "
                "to Terminus2 summarization path for this step."
            )
            return await super()._query_llm(
                chat=chat,
                prompt=prompt,
                logging_paths=logging_paths,
                original_instruction=original_instruction,
                session=session,
            )
        except OutputLengthExceededError:
            self.logger.debug(
                "Output length exceeded during uncertainty sampling; falling back "
                "to Terminus2 output-limit handling for this step."
            )
            return await super()._query_llm(
                chat=chat,
                prompt=prompt,
                logging_paths=logging_paths,
                original_instruction=original_instruction,
                session=session,
            )

        selected = self._select_response(scored)
        self._commit_selected_response(chat, prompt, selected.response, scored)

        if response_path is not None:
            response_path.write_text(selected.response.content)

        if self._selection_debug and logging_path is not None:
            self._write_selection_debug(
                logging_path=logging_path,
                prompt=prompt,
                selected=selected,
                scored=scored,
            )

        return selected.response

    def _score_response(
        self,
        response: LLMResponse,
        prompt: str,
        original_instruction: str,
    ) -> tuple[Optional[float], dict[str, Any]]:
        entropy = _entropy_from_logprobs(response.logprobs)

        if self._uncertainty_method == "entropy":
            return entropy, {"entropy": entropy}

        if self._uncertainty_method == "tracer":
            return entropy, {"tracer_score": entropy, "entropy": entropy}

        if self._uncertainty_method == "saup":
            saup = None if entropy is None else float(math.sqrt(entropy * entropy))
            return saup, {"saup_score": saup, "entropy": entropy}

        if self._uncertainty_method == "trajectory_tau":
            if self._trajectory_tau_module is None:
                raise RuntimeError("trajectory_tau module was not loaded")
            steps = [self._step_to_dict(step) for step in self._trajectory_steps]
            steps.append(
                {
                    "source": "agent",
                    "role": "assistant",
                    "message": response.content,
                    "reasoning_content": response.reasoning_content,
                    "metrics": {
                        "logprobs": response.logprobs or [],
                    },
                }
            )
            result = self._trajectory_tau_module.calculate_trajectory_tau_score(
                steps,
                goal_text=original_instruction or prompt,
                config=self._trajectory_tau_config,
            )
            score = _finite_float(result.get("tau_score"))
            result["entropy"] = entropy
            return score, result

        raise ValueError(f"Unknown uncertainty_method: {self._uncertainty_method}")

    def _select_response(self, scored: list[ScoredResponse]) -> ScoredResponse:
        scoreable = [item for item in scored if item.scoreable]
        if scoreable:
            return min(scoreable, key=lambda item: (item.uncertainty, item.sample_id))
        return scored[0]

    def _commit_selected_response(
        self,
        chat: Chat,
        prompt: str,
        response: LLMResponse,
        scored: list[ScoredResponse],
    ) -> None:
        if response.response_id is not None:
            chat._last_response_id = response.response_id

        for item in scored:
            usage = item.response.usage
            if usage is not None:
                chat._cumulative_input_tokens += usage.prompt_tokens
                chat._cumulative_output_tokens += usage.completion_tokens
                chat._cumulative_cache_tokens += usage.cache_tokens
                chat._cumulative_cost += usage.cost_usd

        chat._accumulate_rollout_details(response)

        assistant_message = {"role": "assistant", "content": response.content}
        if self._interleaved_thinking and response.reasoning_content:
            assistant_message["reasoning_content"] = response.reasoning_content

        chat._messages.extend(
            [
                {"role": "user", "content": prompt},
                assistant_message,
            ]
        )

    def _write_selection_debug(
        self,
        logging_path: Path,
        prompt: str,
        selected: ScoredResponse,
        scored: list[ScoredResponse],
    ) -> None:
        payload = {
            "agent": self.name(),
            "uncertainty_method": self._uncertainty_method,
            "num_samples": self._num_samples,
            "selected_sample_id": selected.sample_id,
            "selected_uncertainty": selected.uncertainty,
            "prompt": prompt,
            "candidates": [
                {
                    "sample_id": item.sample_id,
                    "uncertainty": item.uncertainty,
                    "elapsed_ms": item.elapsed_ms,
                    "n_logprobs": len(item.response.logprobs or []),
                    "usage": _usage_to_dict(item.response.usage),
                    "metrics": item.metrics,
                    "response": item.response.content,
                }
                for item in scored
            ],
        }
        logging_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @staticmethod
    def _step_to_dict(step: Any) -> dict[str, Any]:
        if isinstance(step, dict):
            return step
        if hasattr(step, "model_dump"):
            return step.model_dump(mode="json", exclude_none=True)
        if hasattr(step, "dict"):
            return step.dict(exclude_none=True)
        return dict(step)
