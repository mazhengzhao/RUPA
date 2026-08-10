"""
Trajectory-aware TAU metrics.

This module provides a stronger alternative to the baseline TAU implementation
in ``uncertainty_prop.py``. The original historical influence term relies on a
single semantic/text distance. Here we model uncertainty propagation through
trajectory state:

1. Uncertainty momentum:
   High recent uncertainty should keep affecting later steps.
2. Tool repetition pressure:
   Repeated identical tool calls are a strong signal of unresolved failure.
3. Observation instability:
   Empty/error/stagnant observations propagate uncertainty forward.
4. Action stagnation:
   Repeating the same action mode without new evidence is risky.
5. Goal coverage gap:
   Lightweight coverage check against the task goal and latest user turn.
6. Trajectory subgraph propagation:
   User turns, agent reasoning/actions, and environment feedback are maintained
   as a causal subgraph. Dependency edges propagate uncertainty from relevant
   historical nodes into the current agent node.
7. Logical relation edges:
   Progressive, repetitive, parallel, and feedback-response relations are
   inferred from reasoning/action traces and used as typed graph edges.

The final score keeps the TAU shape:

    risk_i = U_i * (1 + alpha * H_i + beta * C_i)

but replaces H_i with a trajectory-aware propagation model instead of plain
text similarity.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


STOP_WORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "or", "and", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her",
    "am", "can", "will", "would", "could", "should", "may",
    "have", "has", "had", "do", "does", "did",
    "please", "sorry", "wait", "help", "need", "want",
}

ERROR_HINTS = (
    "traceback",
    "error",
    "exception",
    "failed",
    "not found",
    "no such file",
    "module not found",
    "permission denied",
    "timeout",
    "cancelled",
)

PROGRESSION_HINTS = (
    "then",
    "next",
    "now",
    "after",
    "based on",
    "therefore",
    "so",
    "because",
    "continue",
    "proceed",
    "fix",
    "update",
    "modify",
    "verify",
    "test",
)

PARALLEL_HINTS = (
    "alternative",
    "instead",
    "another",
    "different",
    "try",
    "also",
    "separately",
    "meanwhile",
    "option",
    "fallback",
)


@dataclass
class TrajectoryTAUConfig:
    """
    Configuration for trajectory-aware TAU.

    alpha and beta preserve the original TAU structure:
        risk_i = U_i * (1 + alpha * H_i + beta * C_i)

    The remaining weights control the decomposition of H_i.
    """

    alpha: float = 1.0
    beta: float = 0.7
    momentum_weight: float = 0.30
    repetition_weight: float = 0.30
    observation_weight: float = 0.25
    stagnation_weight: float = 0.15
    interaction_goal_weight: float = 0.55
    interaction_user_weight: float = 0.45
    history_decay: float = 0.80
    uncertainty_decay: float = 0.75
    recent_window: int = 5
    novelty_window: int = 4
    graph_weight: float = 0.30
    graph_uncertainty_weight: float = 0.35
    graph_edge_decay: float = 0.85
    graph_max_neighbors: int = 8


@dataclass
class TrajectoryStepState:
    role: str
    text: str
    ui: float
    tool_signatures: list[str]
    observation_text: str
    observation_quality: float
    action_tokens: set[str]


@dataclass
class TrajectoryGraphNode:
    index: int
    state: TrajectoryStepState
    propagated_uncertainty: float


@dataclass
class TrajectoryGraphEdge:
    source_index: int
    target_index: int
    weight: float
    edge_type: str


def _normalize_config(config: Optional[Any]) -> TrajectoryTAUConfig:
    default = TrajectoryTAUConfig()
    if config is None:
        return default

    values = {}
    for field_name in default.__dataclass_fields__:
        if isinstance(config, dict):
            values[field_name] = config.get(field_name, getattr(default, field_name))
        else:
            values[field_name] = getattr(config, field_name, getattr(default, field_name))
    return TrajectoryTAUConfig(**values)


def _safe_get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_role(raw_role: Any) -> str:
    if raw_role in ("assistant", "agent"):
        return "assistant"
    if raw_role == "user":
        return "user"
    if raw_role in ("environment", "env", "tool", "observation", "terminal", "system"):
        return "environment"
    return str(raw_role or "")


def _clean_token(token: str) -> str:
    return token.strip('.,!?;:()[]{}"\'-').lower()


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()

    tokens = set()
    for raw in text.split():
        cleaned = _clean_token(raw)
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace(".", "").replace("-", "").isdigit():
            continue
        tokens.add(cleaned)
    return tokens


def _extract_text(msg: Any) -> str:
    parts: list[str] = []

    content = _safe_get_attr(msg, "content")
    if content is None:
        content = _safe_get_attr(msg, "message", "")
    if isinstance(content, str) and content.strip():
        parts.append(content)
    elif content:
        parts.append(str(content))

    reasoning_content = _safe_get_attr(msg, "reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        parts.append(reasoning_content)
    elif reasoning_content:
        parts.append(str(reasoning_content))

    observation_text = _extract_observation_text(msg)
    if observation_text.strip():
        parts.append(observation_text)

    return "\n".join(parts)


def _extract_observation_text(msg: Any) -> str:
    observation = _safe_get_attr(msg, "observation")
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    return json.dumps(observation, ensure_ascii=False, sort_keys=True)


def _tool_signature(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        name = tool_call.get("function_name") or tool_call.get("name") or ""
        arguments = tool_call.get("arguments", {})
    else:
        name = getattr(tool_call, "function_name", "") or getattr(tool_call, "name", "")
        arguments = getattr(tool_call, "arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass

    if isinstance(arguments, dict):
        canonical_args = sorted(arguments.items())
    else:
        canonical_args = [str(arguments)]

    return f"{name}:{canonical_args}"


def _extract_tool_signatures(msg: Any) -> list[str]:
    tool_calls = _safe_get_attr(msg, "tool_calls", []) or []
    signatures = []
    for tool_call in tool_calls:
        sig = _tool_signature(tool_call)
        if sig:
            signatures.append(sig)
    return signatures


def _extract_uncertainty(msg: Any) -> float:
    uncertainty = _safe_get_attr(msg, "uncertainty")
    if isinstance(uncertainty, dict):
        value = uncertainty.get("normalized_entropy")
        if value is not None:
            return float(value)
    metrics = _safe_get_attr(msg, "metrics")
    if isinstance(metrics, dict):
        logprobs = metrics.get("logprobs", [])
        cleaned = []
        for item in logprobs:
            try:
                item = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(item):
                cleaned.append(item)
        if cleaned:
            return float(-np.mean(cleaned))
    return 0.0


def _observation_quality(observation_text: str) -> float:
    """
    Estimate how unresolved the most recent observation is.

    Higher value means worse quality / stronger uncertainty propagation.
    """

    if not observation_text.strip():
        return 1.0

    text = observation_text.lower()
    if any(hint in text for hint in ERROR_HINTS):
        return 1.0

    tokens = _tokenize(observation_text)
    if not tokens:
        return 0.8

    return 0.0


def _build_state(msg: Any) -> TrajectoryStepState:
    role = _normalize_role(_safe_get_attr(msg, "role") or _safe_get_attr(msg, "source"))
    text = _extract_text(msg)
    observation_text = _extract_observation_text(msg)
    tool_signatures = _extract_tool_signatures(msg)
    action_tokens = _tokenize(text) | _tokenize(observation_text)
    for sig in tool_signatures:
        action_tokens |= _tokenize(sig)

    return TrajectoryStepState(
        role=role,
        text=text,
        ui=_extract_uncertainty(msg),
        tool_signatures=tool_signatures,
        observation_text=observation_text,
        observation_quality=_observation_quality(observation_text),
        action_tokens=action_tokens,
    )


def _weighted_mean(values: list[float], decay: float) -> float:
    if not values:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0
    for i, value in enumerate(reversed(values)):
        weight = decay ** i
        weighted_sum += weight * value
        weight_sum += weight

    if weight_sum == 0.0:
        return 0.0
    return weighted_sum / weight_sum


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return float(len(tokens_a & tokens_b) / len(union))


def safe_mean(values: list[float]) -> Optional[float]:
    cleaned = [x for x in values if math.isfinite(x)]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _edge_strength(edge_type: str, raw_weight: float, age: int, config: TrajectoryTAUConfig) -> float:
    if raw_weight <= 0.0:
        return 0.0
    decayed = raw_weight * (config.graph_edge_decay ** max(0, age - 1))
    caps = {
        "sequential": 0.35,
        "latest_user": 0.75,
        "tool_repetition": 0.95,
        "token_overlap": 0.70,
        "feedback_instability": 0.85,
        "progression": 0.65,
        "repetition": 0.95,
        "parallel": 0.45,
        "feedback_response": 0.90,
    }
    return float(min(caps.get(edge_type, 1.0), decayed))


def _contains_hint(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _same_action_mode(current: TrajectoryStepState, source: TrajectoryStepState) -> bool:
    return bool(current.tool_signatures) == bool(source.tool_signatures)


def _classify_logical_relation(
    current: TrajectoryStepState,
    source: TrajectoryStepState,
    token_overlap: float,
    tool_overlap: float,
) -> tuple[str, float]:
    """
    Infer a logical relation from a historical node to the current agent node.

    Relation semantics:
    - repetition: same tool/action surface, little novelty; likely stuck.
    - feedback_response: current step reacts to a bad or empty observation.
    - progression: current step continues/refines a previous line of work.
    - parallel: current step explores an alternative branch with weak overlap.
    """

    current_text = current.text.lower()
    source_text = source.text.lower()
    combined_text = f"{source_text}\n{current_text}"

    if source.role in {"assistant", "environment"} and source.observation_quality > 0.0:
        response_strength = max(source.observation_quality, token_overlap)
        if response_strength >= 0.25 or _contains_hint(current_text, PROGRESSION_HINTS):
            return "feedback_response", min(1.0, 0.55 + 0.45 * response_strength)

    if tool_overlap >= 0.80 or (token_overlap >= 0.65 and _same_action_mode(current, source)):
        return "repetition", max(tool_overlap, token_overlap)

    if _contains_hint(current_text, PARALLEL_HINTS) and token_overlap < 0.45:
        return "parallel", max(0.25, 1.0 - token_overlap)

    if token_overlap >= 0.20 and (
        _contains_hint(combined_text, PROGRESSION_HINTS)
        or _same_action_mode(current, source)
        or source.role == "user"
    ):
        return "progression", min(1.0, 0.35 + 0.65 * token_overlap)

    if token_overlap >= 0.15:
        return "token_overlap", token_overlap

    return "", 0.0


def build_dependency_edges(
    current: TrajectoryStepState,
    graph_nodes: list[TrajectoryGraphNode],
    latest_user_index: Optional[int],
    config: TrajectoryTAUConfig,
) -> list[TrajectoryGraphEdge]:
    """
    Add dependency edges from historical graph nodes into the current node.

    Edges are intentionally lightweight and local. In addition to causal
    adjacency and user-to-agent dependency, they infer logical relations:
    progression, repetition, parallel exploration, and feedback response.
    """

    if not graph_nodes:
        return []

    target_index = len(graph_nodes)
    edges: list[TrajectoryGraphEdge] = []

    previous = graph_nodes[-1]
    edges.append(
        TrajectoryGraphEdge(
            source_index=previous.index,
            target_index=target_index,
            weight=_edge_strength("sequential", 0.35, 1, config),
            edge_type="sequential",
        )
    )

    if latest_user_index is not None:
        user_age = target_index - latest_user_index
        edges.append(
            TrajectoryGraphEdge(
                source_index=latest_user_index,
                target_index=target_index,
                weight=_edge_strength("latest_user", 0.75, user_age, config),
                edge_type="latest_user",
            )
        )

    recent_nodes = graph_nodes[-max(1, config.graph_max_neighbors):]
    current_tools = set(current.tool_signatures)
    latest_user_tokens = set()
    if latest_user_index is not None:
        for node in graph_nodes:
            if node.index == latest_user_index:
                latest_user_tokens = node.state.action_tokens
                break

    for node in recent_nodes:
        source = node.state
        age = target_index - node.index
        source_tools = set(source.tool_signatures)
        tool_overlap = _jaccard_similarity(current_tools, source_tools)
        token_overlap = _jaccard_similarity(current.action_tokens, source.action_tokens)

        if current_tools and source_tools and tool_overlap > 0.0:
            edges.append(
                TrajectoryGraphEdge(
                    source_index=node.index,
                    target_index=target_index,
                    weight=_edge_strength("tool_repetition", tool_overlap, age, config),
                    edge_type="tool_repetition",
                )
            )

        relation_type, relation_weight = _classify_logical_relation(
            current,
            source,
            token_overlap,
            tool_overlap,
        )

        if not relation_type and latest_user_tokens and source.role == "assistant":
            current_user_overlap = _jaccard_similarity(current.action_tokens, latest_user_tokens)
            source_user_overlap = _jaccard_similarity(source.action_tokens, latest_user_tokens)
            if (
                current_user_overlap >= 0.05
                and source_user_overlap >= 0.05
                and token_overlap < 0.35
                and not _same_action_mode(current, source)
            ):
                relation_type = "parallel"
                relation_weight = 0.25 + 0.5 * min(current_user_overlap, source_user_overlap)

        if relation_type:
            edges.append(
                TrajectoryGraphEdge(
                    source_index=node.index,
                    target_index=target_index,
                    weight=_edge_strength(relation_type, relation_weight, age, config),
                    edge_type=relation_type,
                )
            )

        if token_overlap >= 0.15:
            edges.append(
                TrajectoryGraphEdge(
                    source_index=node.index,
                    target_index=target_index,
                    weight=_edge_strength("token_overlap", token_overlap, age, config),
                    edge_type="token_overlap",
                )
            )

        if source.role in {"assistant", "environment"} and source.observation_quality > 0.0:
            edges.append(
                TrajectoryGraphEdge(
                    source_index=node.index,
                    target_index=target_index,
                    weight=_edge_strength("feedback_instability", source.observation_quality, age, config),
                    edge_type="feedback_instability",
                )
            )
            if token_overlap >= 0.10 or source.role == "environment":
                edges.append(
                    TrajectoryGraphEdge(
                        source_index=node.index,
                        target_index=target_index,
                        weight=_edge_strength("feedback_response", max(source.observation_quality, token_overlap), age, config),
                        edge_type="feedback_response",
                    )
                )

    deduped: dict[tuple[int, str], TrajectoryGraphEdge] = {}
    for edge in edges:
        if edge.weight <= 0.0:
            continue
        key = (edge.source_index, edge.edge_type)
        previous_edge = deduped.get(key)
        if previous_edge is None or edge.weight > previous_edge.weight:
            deduped[key] = edge

    edges = list(deduped.values())
    edges.sort(key=lambda edge: edge.weight, reverse=True)
    return edges[: max(1, config.graph_max_neighbors)]


def calculate_graph_uncertainty(
    current: TrajectoryStepState,
    graph_nodes: list[TrajectoryGraphNode],
    latest_user_index: Optional[int],
    config: TrajectoryTAUConfig,
) -> dict[str, Any]:
    edges = build_dependency_edges(current, graph_nodes, latest_user_index, config)
    if not edges:
        return {
            "graph_uncertainty": 0.0,
            "graph_edge_count": 0,
            "graph_total_weight": 0.0,
            "graph_max_edge_weight": 0.0,
            "graph_edges": [],
        }

    node_by_index = {node.index: node for node in graph_nodes}
    weighted_uncertainty = 0.0
    total_weight = 0.0
    edge_payloads = []

    for edge in edges:
        source = node_by_index.get(edge.source_index)
        if source is None:
            continue
        weighted_uncertainty += edge.weight * source.propagated_uncertainty
        total_weight += edge.weight
        edge_payloads.append(
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "weight": edge.weight,
                "edge_type": edge.edge_type,
                "source_role": source.state.role,
                "source_uncertainty": source.propagated_uncertainty,
            }
        )

    if total_weight == 0.0:
        graph_uncertainty = 0.0
    else:
        graph_uncertainty = weighted_uncertainty / total_weight

    return {
        "graph_uncertainty": float(max(0.0, min(1.5, graph_uncertainty))),
        "graph_edge_count": len(edge_payloads),
        "graph_total_weight": float(total_weight),
        "graph_max_edge_weight": float(max(edge["weight"] for edge in edge_payloads) if edge_payloads else 0.0),
        "graph_edges": edge_payloads,
    }


def calculate_uncertainty_momentum(history: list[TrajectoryStepState], decay: float) -> float:
    agent_history = [state.ui for state in history if state.role == "assistant"]
    if not agent_history:
        return 0.0

    weighted_level = _weighted_mean(agent_history, decay)
    if len(agent_history) == 1:
        return min(1.0, weighted_level)

    diffs = [max(0.0, agent_history[i] - agent_history[i - 1]) for i in range(1, len(agent_history))]
    weighted_diff = _weighted_mean(diffs, decay)
    momentum = 0.7 * weighted_level + 0.3 * weighted_diff
    return float(max(0.0, min(1.0, momentum)))


def calculate_tool_repetition_pressure(current: TrajectoryStepState, history: list[TrajectoryStepState], recent_window: int) -> float:
    if not current.tool_signatures:
        return 0.0

    recent_history = [state for state in history if state.role == "assistant"][-recent_window:]
    if not recent_history:
        return 0.0

    overlap_scores = []
    current_set = set(current.tool_signatures)
    for state in recent_history:
        if not state.tool_signatures:
            continue
        past_set = set(state.tool_signatures)
        union = current_set | past_set
        if not union:
            continue
        overlap_scores.append(len(current_set & past_set) / len(union))

    if not overlap_scores:
        return 0.0
    return float(max(overlap_scores))


def calculate_observation_instability(history: list[TrajectoryStepState], recent_window: int) -> float:
    recent_agent_states = [state for state in history if state.role == "assistant"][-recent_window:]
    if not recent_agent_states:
        return 0.0

    qualities = [state.observation_quality for state in recent_agent_states]
    instability = safe_mean(qualities) or 0.0

    stagnant_bonus = 0.0
    recent_observations = [
        state.observation_text.strip()
        for state in recent_agent_states
        if state.observation_text.strip()
    ]
    if len(recent_observations) >= 2 and len(set(recent_observations[-2:])) == 1:
        stagnant_bonus = 0.25

    return float(min(1.0, instability + stagnant_bonus))


def calculate_action_stagnation(current: TrajectoryStepState, history: list[TrajectoryStepState], novelty_window: int) -> float:
    recent_history = [state for state in history if state.role == "assistant"][-novelty_window:]
    if not recent_history:
        return 0.0

    history_tokens = set()
    for state in recent_history:
        history_tokens |= state.action_tokens

    if not current.action_tokens:
        return 0.0

    new_tokens = current.action_tokens - history_tokens
    novelty = len(new_tokens) / max(1, len(current.action_tokens))
    stagnation = 1.0 - novelty

    same_mode_count = sum(
        1
        for state in recent_history
        if bool(state.tool_signatures) == bool(current.tool_signatures)
    )
    mode_bonus = same_mode_count / max(1, len(recent_history))

    return float(min(1.0, 0.7 * stagnation + 0.3 * mode_bonus))


def calculate_goal_coverage_gap(current: TrajectoryStepState, latest_user: Optional[TrajectoryStepState], goal_text: str, config: TrajectoryTAUConfig) -> float:
    current_tokens = current.action_tokens
    if not current_tokens:
        return 0.0

    def coverage_gap(reference_text: str) -> float:
        ref_tokens = _tokenize(reference_text)
        if not ref_tokens:
            return 0.0
        covered = len(current_tokens & ref_tokens) / len(ref_tokens)
        return 1.0 - covered

    gap = 0.0
    if goal_text:
        gap += config.interaction_goal_weight * coverage_gap(goal_text)
    if latest_user is not None:
        gap += config.interaction_user_weight * coverage_gap(latest_user.text)

    return float(max(0.0, min(1.5, gap)))


def calculate_trajectory_propagation(
    current: TrajectoryStepState,
    history: list[TrajectoryStepState],
    config: TrajectoryTAUConfig,
) -> dict[str, float]:
    """
    Calculate structured uncertainty propagation H_i from trajectory dynamics.
    """

    momentum = calculate_uncertainty_momentum(history, config.uncertainty_decay)
    repetition = calculate_tool_repetition_pressure(current, history, config.recent_window)
    observation = calculate_observation_instability(history, config.recent_window)
    stagnation = calculate_action_stagnation(current, history, config.novelty_window)

    propagation = (
        config.momentum_weight * momentum +
        config.repetition_weight * repetition +
        config.observation_weight * observation +
        config.stagnation_weight * stagnation
    )

    return {
        "propagation": float(max(0.0, min(1.5, propagation))),
        "momentum": momentum,
        "repetition": repetition,
        "observation": observation,
        "stagnation": stagnation,
    }


def calculate_trajectory_tau_score(
    messages: list[Any],
    goal_text: str,
    config: Optional[TrajectoryTAUConfig] = None,
) -> dict[str, Any]:
    """
    Trajectory-aware replacement for baseline TAU.

    Compared to ``calculate_tau_score`` in ``uncertainty_prop.py``, this version
    models H_i with structured trajectory signals rather than a single semantic
    distance between the current text and prior texts.
    """

    config = _normalize_config(config)

    history: list[TrajectoryStepState] = []
    graph_nodes: list[TrajectoryGraphNode] = []
    latest_user: Optional[TrajectoryStepState] = None
    latest_user_index: Optional[int] = None
    risks: list[float] = []
    per_step: list[dict[str, float]] = []

    for msg in messages:
        state = _build_state(msg)
        if state.role not in {"assistant", "user", "environment"}:
            continue

        if state.role == "assistant":
            propagation = calculate_trajectory_propagation(state, history, config)
            graph = calculate_graph_uncertainty(state, graph_nodes, latest_user_index, config)
            interaction_gap = calculate_goal_coverage_gap(state, latest_user, goal_text, config)
            effective_ui = state.ui + config.graph_uncertainty_weight * graph["graph_uncertainty"]
            combined_propagation = min(
                1.5,
                propagation["propagation"] + config.graph_weight * graph["graph_uncertainty"],
            )
            risk = effective_ui * (1.0 + config.alpha * combined_propagation + config.beta * interaction_gap)

            risks.append(float(risk))
            per_step.append(
                {
                    "ui": state.ui,
                    "effective_ui": float(effective_ui),
                    "risk": float(risk),
                    "propagation": propagation["propagation"],
                    "combined_propagation": float(combined_propagation),
                    "graph_uncertainty": graph["graph_uncertainty"],
                    "graph_edge_count": float(graph["graph_edge_count"]),
                    "graph_total_weight": graph["graph_total_weight"],
                    "graph_max_edge_weight": graph["graph_max_edge_weight"],
                    "graph_edges": graph["graph_edges"],
                    "interaction_gap": interaction_gap,
                    "momentum": propagation["momentum"],
                    "repetition": propagation["repetition"],
                    "observation": propagation["observation"],
                    "stagnation": propagation["stagnation"],
                }
            )
            node_uncertainty = float(max(state.ui, effective_ui, risk))
        elif state.role == "environment":
            node_uncertainty = state.observation_quality
        else:
            node_uncertainty = state.observation_quality

        graph_nodes.append(
            TrajectoryGraphNode(
                index=len(graph_nodes),
                state=state,
                propagated_uncertainty=float(max(0.0, min(1.5, node_uncertainty))),
            )
        )
        history.append(state)
        if state.role == "user":
            latest_user = state
            latest_user_index = graph_nodes[-1].index

    if not risks:
        return {"tau_score": 0.0, "num_steps": 0, "per_step": []}

    tau = float(np.mean(risks))
    return {
        "tau_score": tau,
        "tau_confidence": float(1.0 / (1.0 + tau)),
        "num_steps": len(risks),
        "mean_risk": float(np.mean(risks)),
        "max_risk": float(np.max(risks)),
        "mean_propagation": float(np.mean([step["propagation"] for step in per_step])),
        "mean_combined_propagation": float(np.mean([step["combined_propagation"] for step in per_step])),
        "mean_graph_uncertainty": float(np.mean([step["graph_uncertainty"] for step in per_step])),
        "mean_effective_ui": float(np.mean([step["effective_ui"] for step in per_step])),
        "mean_interaction_gap": float(np.mean([step["interaction_gap"] for step in per_step])),
        "per_step": per_step,
    }
