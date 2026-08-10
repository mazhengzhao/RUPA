"""
Uncertainty Analysis Script

Analyzes Tau-2 simulation trajectories and calculates uncertainty metrics
using the TRACER framework.

By default, results are saved to data/uncertainty/ with the same filename as the input.

Usage:
    # Basic analysis (auto-saves to data/uncertainty/)
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json
    
    # Detailed turn-by-turn view
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --detailed
    
    # Custom output location
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --output results/my_analysis.json
    
    # Don't save, just display
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --no-save

Example:
    Input:  data/simulations/2025-11-06_airline_gemini.json
    Output: data/uncertainty/2025-11-06_airline_gemini.json (auto-saved)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from tau2.data_model.simulation import Results
from tau2.metrics.uncertainty import (
    TRACERConfig,
    calculate_normalized_entropy,
    calculate_saup_score,
    calculate_tracer_score,
    get_uncertainty_stats,
    calculate_hybrid_repetition_score,
    calculate_tool_repetition
)

# Optional sklearn import for AUROC calculation
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. AUROC evaluation will be skipped.")


class TurnUncertainty(BaseModel):
    """Uncertainty information for a single turn."""

    turn: int
    actor: str
    role: str
    content_preview: str
    statistics: Optional[dict] = None
    da_score: Optional[float] = None
    do_score: Optional[float] = None
    do_type: Optional[str] = None
    normentropy_filtered_score: float  # Normalized entropy with content-token filtering applied
    normentropy_score: Optional[float] = None  # Raw normalized entropy (before filtering)
    selfconf_score: Optional[float] = None  # Self-assessed confidence from raw data
    saup_score: Optional[float] = None  # Per-turn cumulative SAUP score
    tracer_score: Optional[float] = None  # Per-turn cumulative TRACER score


class SimulationUncertainty(BaseModel):
    """Uncertainty analysis for a single simulation."""

    simulation_id: str
    task_id: str
    trial: int
    turn_count: int
    uncertainty_scores: list[TurnUncertainty]
    summary: dict
    tracer_metrics: Optional[dict] = None
    ground_truth_pass: Optional[bool] = None


class AUROCMetrics(BaseModel):
    """AUROC evaluation metrics for failure prediction."""
    
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    optimal_threshold: float
    num_samples: int
    num_failures: int
    num_successes: int
    mean_tracer_failures: float
    mean_tracer_successes: float
    std_tracer_failures: float
    std_tracer_successes: float


class AUARCMetrics(BaseModel):
    """AUARC evaluation metrics for selective prediction.
    
    Area Under Accuracy-Rejection Curve measures how well the uncertainty
    score can identify which predictions to reject to improve accuracy.
    
    The curve plots accuracy vs. coverage (1 - rejection_rate), where we
    progressively reject samples with highest uncertainty scores.
    """
    
    auarc: float  # Area under the accuracy-rejection curve (higher is better)
    max_accuracy: float  # Maximum achievable accuracy with selective prediction
    accuracy_at_50_coverage: float  # Accuracy when covering 50% of samples
    accuracy_at_80_coverage: float  # Accuracy when covering 80% of samples
    optimal_coverage: float  # Coverage that maximizes (accuracy - baseline_accuracy)
    optimal_accuracy: float  # Accuracy at optimal coverage
    baseline_accuracy: float  # Accuracy without rejection (all samples)
    num_samples: int
    num_failures: int
    num_successes: int
    coverages: list[float]  # Coverage points (for plotting)
    accuracies: list[float]  # Accuracy at each coverage (for plotting)


class EarlyWarningMetrics(BaseModel):
    """Early warning analysis metrics showing when each metric signals failure."""
    
    num_failed_tasks: int
    metric_name: str
    mean_peak_percentage: float  # Mean percentage of trajectory where peak occurs
    median_peak_percentage: float  # Median percentage of trajectory where peak occurs
    std_peak_percentage: float  # Std dev of peak percentages
    percent_before_10: float  # Percentage of tasks where metric crosses before 10% of turns
    percent_before_20: float  # Percentage of tasks where metric crosses before 20% of turns
    percent_before_30: float  # Percentage of tasks where metric crosses before 30% of turns
    percent_before_40: float  # Percentage of tasks where metric crosses before 40% of turns
    percent_before_50: float  # Percentage of tasks where metric crosses before 50% of turns
    percent_before_60: float  # Percentage of tasks where metric crosses before 60% of turns
    percent_before_70: float  # Percentage of tasks where metric crosses before 70% of turns
    percent_before_80: float  # Percentage of tasks where metric crosses before 80% of turns
    percent_before_90: float  # Percentage of tasks where metric crosses before 90% of turns


class UncertaintyAnalysis(BaseModel):
    """Complete uncertainty analysis results."""

    metadata: dict
    results: list[SimulationUncertainty]
    auroc_metrics: Optional[AUROCMetrics] = None
    auarc_metrics: Optional[AUARCMetrics] = None  # NEW: Selective prediction metrics for TRACER
    baseline_aurocs: Optional[dict] = None
    baseline_auarcs: Optional[dict] = None  # NEW: Selective prediction metrics for baselines
    ablation_studies: Optional[dict] = None  # Stores ablation analysis results
    early_warning_metrics: Optional[dict[str, EarlyWarningMetrics]] = None  # NEW: Early warning analysis


def analyze_simulation(simulation: dict, config: TRACERConfig, verbose: bool = False) -> SimulationUncertainty:
    """
    Analyze a single simulation and extract uncertainty scores.

    This function recalculates repetition metrics using the Hybrid Repetition approach
    (combining semantic and lexical similarity) to distinguish enumeration from looping.
    It also applies strict tool repetition penalties for exact duplicate tool calls.

    Args:
        simulation: A simulation run dictionary
        config: TRACER configuration for weighted aggregation
        verbose: If True, include detailed statistics

    Returns:
        SimulationUncertainty: Analyzed uncertainty data
    """
    uncertainty_scores = []
    turn_counter = 0
    
    # Track agent history for repetition detection
    agent_text_history = []  # Text responses for hybrid repetition
    agent_tool_history = []  # Tool calls for exact duplicate detection

    for message in simulation.get("messages", []):
        role = message.get("role")

        # Only process agent and user messages
        if role not in ["assistant", "user"]:
            continue

        turn_counter += 1

        # Determine actor type
        actor = "agent" if role == "assistant" else "user"

        # Extract logprobs and calculate filtered normalized entropy
        logprobs = message.get("logprobs")
        normentropy_filtered_score = calculate_normalized_entropy(logprobs)
        
        # Extract raw normalized entropy (before filtering) from uncertainty data
        normentropy_score = None
        selfconf_score = None
        if "uncertainty" in message and message["uncertainty"] is not None:
            normentropy_score = message["uncertainty"].get("normalized_entropy")
            selfconf_score = message["uncertainty"].get("self_assessed_confidence")
        
        # Calculate HYBRID REPETITION for agent messages
        da_score = None
        if role == "assistant":
            # Extract content
            content = message.get("content", "")
            if content is None:
                content = ""
            
            # Calculate text-based hybrid repetition
            text_repetition = 0.0
            if content:
                text_repetition = calculate_hybrid_repetition_score(
                    content, 
                    agent_text_history
                )
                # Update text history
                agent_text_history.append(content)
            
            # Calculate tool-based repetition
            tool_repetition = 0.0
            tool_calls = message.get("tool_calls")
            if tool_calls:
                tool_repetition = calculate_tool_repetition(
                    tool_calls,
                    agent_tool_history
                )
                # Update tool history
                agent_tool_history.append(tool_calls)
            
            # Aggregate: take maximum (worst-case penalty)
            # Either text looping OR tool duplication is a failure signal
            da_score = max(text_repetition, tool_repetition)
            
            logger.debug(
                f"Turn {turn_counter}: text_rep={text_repetition:.3f}, "
                f"tool_rep={tool_repetition:.3f}, final_da={da_score:.3f}"
            )
        
        # Extract inference gap (Do) metrics from pre-computed data
        do_score = message.get("do_score")
        do_type = message.get("do_type")

        # Build turn data
        turn_data = TurnUncertainty(
            turn=turn_counter,
            actor=actor,
            role=role,
            content_preview=(
                message.get("content", "")[:100]
                if message.get("content")
                else "[tool_call]"
            ),
            da_score=da_score,  # Hybrid Repetition (text + tool)
            do_score=do_score,
            do_type=do_type,
            # Baseline and TRACER metrics
            normentropy_filtered_score=normentropy_filtered_score,
            normentropy_score=normentropy_score,
            selfconf_score=selfconf_score,
        )

        # Add detailed statistics if requested
        if verbose and logprobs is not None:
            stats = get_uncertainty_stats(logprobs)
            turn_data.statistics = stats.model_dump()

        uncertainty_scores.append(turn_data)
    
    # Calculate cumulative TRACER and SAUP at each turn (for early warning analysis)
    for idx in range(len(uncertainty_scores)):
        # Build step_data up to current turn (cumulative)
        cumulative_step_data = []
        for i in range(idx + 1):
            turn = uncertainty_scores[i]
            cumulative_step_data.append({
                "ui": turn.normentropy_filtered_score,
                "da": turn.da_score,
                "do_agent": turn.do_score if turn.do_type == "agent_coherence" else None,
                "do_user": turn.do_score if turn.do_type == "user_coherence" else None
            })
        
        # Calculate cumulative TRACER
        cumulative_tracer_result = calculate_tracer_score(cumulative_step_data, config)
        uncertainty_scores[idx].tracer_score = cumulative_tracer_result['tracer_score']
        
        # Calculate cumulative SAUP
        cumulative_saup_result = calculate_saup_score(cumulative_step_data, config)
        uncertainty_scores[idx].saup_score = cumulative_saup_result['saup_score']

    # Calculate TRACER aggregation score (full trajectory)
    tracer_metrics = None
    if uncertainty_scores:
        step_data = [
            {
                "ui": turn.normentropy_filtered_score,
                "da": turn.da_score,  # Hybrid repetition score
                "do_agent": turn.do_score if turn.do_type == "agent_coherence" else None,
                "do_user": turn.do_score if turn.do_type == "user_coherence" else None
            }
            for turn in uncertainty_scores
        ]
        tracer_result = calculate_tracer_score(step_data, config)
        # Remove penalties list (too verbose)
        tracer_metrics = {k: v for k, v in tracer_result.items() if k != 'penalties'}
    
    # Extract ground truth (task success)
    ground_truth = simulation.get("reward_info", {}).get("reward", None) if simulation.get("reward_info") else None
    ground_truth_pass = ground_truth == 1.0 if ground_truth is not None else None

    # Calculate summary statistics
    summary = {}
    if uncertainty_scores:
        agent_scores = [s.normentropy_filtered_score for s in uncertainty_scores if s.actor == "agent"]
        user_scores = [s.normentropy_filtered_score for s in uncertainty_scores if s.actor == "user"]
        
        # Repetition and coherence metrics
        repetition_scores = [s.da_score for s in uncertainty_scores if s.da_score is not None]
        do_scores = [s.do_score for s in uncertainty_scores if s.do_score is not None]
        do_agent_coherence = [s.do_score for s in uncertainty_scores if s.do_score is not None and s.do_type == "agent_coherence"]
        do_user_coherence = [s.do_score for s in uncertainty_scores if s.do_score is not None and s.do_type == "user_coherence"]

        summary = {
            "mean_uncertainty_overall": float(
                np.mean([s.normentropy_filtered_score for s in uncertainty_scores])
            ),
            "mean_uncertainty_agent": float(np.mean(agent_scores)) if agent_scores else 0.0,
            "mean_uncertainty_user": float(np.mean(user_scores)) if user_scores else 0.0,
            "max_uncertainty_overall": float(
                np.max([s.normentropy_filtered_score for s in uncertainty_scores])
            ),
            "agent_turn_count": len(agent_scores),
            "user_turn_count": len(user_scores),
            # Hybrid repetition metrics
            "mean_repetition_score": float(np.mean(repetition_scores)) if repetition_scores else None,
            "std_repetition_score": float(np.std(repetition_scores)) if repetition_scores else None,
            "mean_do_score": float(np.mean(do_scores)) if do_scores else None,
            "std_do_score": float(np.std(do_scores)) if do_scores else None,
            "mean_do_agent_coherence": float(np.mean(do_agent_coherence)) if do_agent_coherence else None,
            "mean_do_user_coherence": float(np.mean(do_user_coherence)) if do_user_coherence else None,
            "repetition_count": len(repetition_scores),
            "do_count": len(do_scores),
        }

    return SimulationUncertainty(
        simulation_id=simulation.get("id", "unknown"),
        task_id=simulation.get("task_id", "unknown"),
        trial=simulation.get("trial", 0),
        turn_count=len(uncertainty_scores),
        uncertainty_scores=uncertainty_scores,
        summary=summary,
        tracer_metrics=tracer_metrics,
        ground_truth_pass=ground_truth_pass,
    )


def calculate_auroc_metrics(
    analyzed_sims: list[SimulationUncertainty], 
    role_filter: Optional[str] = None,
    config: Optional[TRACERConfig] = None
) -> Optional[AUROCMetrics]:
    """
    Calculate AUROC metrics for TRACER failure prediction.
    
    Hypothesis: High TRACER score predicts task failure.
    Label encoding: Failure=1, Success=0
    
    Args:
        analyzed_sims: List of analyzed simulations with TRACER metrics
        role_filter: Optional role filter ("assistant" or "user"). If None, uses all turns.
        config: TRACER configuration (only needed if role_filter is specified)
        
    Returns:
        AUROCMetrics if calculation successful, None otherwise
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Skipping AUROC calculation.")
        return None
    
    # Extract TRACER scores and ground truth labels
    y_scores = []
    y_true = []
    
    for sim in analyzed_sims:
        # Need ground truth
        if sim.ground_truth_pass is None:
            continue
        
        # If role_filter is specified, recalculate TRACER with filtered turns
        if role_filter is not None:
            if config is None:
                config = TRACERConfig()
            
            # Filter uncertainty scores by role
            # Map role to actor: "assistant" -> "agent", "user" -> "user"
            target_actor = "agent" if role_filter == "assistant" else "user"
            filtered_scores = [s for s in sim.uncertainty_scores if s.actor == target_actor]
            
            if not filtered_scores:
                continue
            
            # Build step_data from filtered scores
            step_data = [
                {
                    "ui": turn.normentropy_filtered_score,
                    "da": turn.da_score,
                    "do_agent": turn.do_score if turn.do_type == "agent_coherence" else None,
                    "do_user": turn.do_score if turn.do_type == "user_coherence" else None
                }
                for turn in filtered_scores
            ]
            
            tracer_result = calculate_tracer_score(step_data, config)
            tracer_score = tracer_result.get('tracer_score')
        else:
            # Use pre-calculated TRACER score from full analysis
            if sim.tracer_metrics is None:
                continue
            tracer_score = sim.tracer_metrics.get('tracer_score')
        
        if tracer_score is None:
            continue
        
        # Label encoding: Failure=1, Success=0
        ground_truth = 0 if sim.ground_truth_pass else 1
        
        y_scores.append(tracer_score)
        y_true.append(ground_truth)
    
    # Need at least 2 samples and both classes present
    if len(y_scores) < 2:
        logger.warning(f"Not enough samples for AUROC ({len(y_scores)} < 2). Skipping.")
        return None
    
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in data: {unique_labels}. Cannot calculate AUROC.")
        return None
    
    try:
        # Calculate AUROC
        auroc = roc_auc_score(y_true, y_scores)
        
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Statistics by class
        failures = y_scores[y_true == 1]
        successes = y_scores[y_true == 0]
        
        return AUROCMetrics(
            auroc=float(auroc),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            optimal_threshold=float(optimal_threshold),
            num_samples=len(y_scores),
            num_failures=int(np.sum(y_true == 1)),
            num_successes=int(np.sum(y_true == 0)),
            mean_tracer_failures=float(np.mean(failures)) if len(failures) > 0 else 0.0,
            mean_tracer_successes=float(np.mean(successes)) if len(successes) > 0 else 0.0,
            std_tracer_failures=float(np.std(failures)) if len(failures) > 0 else 0.0,
            std_tracer_successes=float(np.std(successes)) if len(successes) > 0 else 0.0,
        )
    except Exception as e:
        logger.error(f"Failed to calculate AUROC: {e}")
        return None


def calculate_baseline_aurocs(
    analyzed_sims: list[SimulationUncertainty], 
    original_results: Results = None,
    role_filter: Optional[str] = None,
    config: Optional[TRACERConfig] = None
) -> Optional[dict]:
    """
    Calculate AUROC metrics for baseline predictors using the same schema as TRACER.
    
    This function evaluates baseline metrics to compare against TRACER:
    1. Normalized Entropy only (mean U_i)
    2. Self-assessed Confidence (extracted from original simulation messages if available)
    3. SAUP (multiplicative weighting with RMS aggregation)
    
    Args:
        analyzed_sims: List of analyzed simulations with summary statistics
        original_results: Original Results object to extract self-assessed confidence
        role_filter: Optional role filter ("assistant" or "user"). If None, uses all turns.
        config: TRACER configuration (only needed for SAUP and if role_filter is specified)
        
    Returns:
        Dictionary with baseline metrics matching TRACER AUROC schema, or None if calculation fails
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Skipping baseline AUROC calculation.")
        return None
    
    if config is None:
        config = TRACERConfig()
    
    results = {}
    
    #########################################################################################
    # Baseline 1: Normalized Entropy (U_i) only
    #########################################################################################
    y_scores_entropy = []
    y_true_entropy = []
    
    for sim in analyzed_sims:
        if sim.ground_truth_pass is None:
            continue
        
        # If role_filter is specified, recalculate mean uncertainty for filtered turns
        if role_filter is not None:
            target_actor = "agent" if role_filter == "assistant" else "user"
            filtered_scores = [s for s in sim.uncertainty_scores if s.actor == target_actor]
            
            if not filtered_scores:
                continue
            
            mean_uncertainty = float(np.mean([s.normentropy_filtered_score for s in filtered_scores]))
        else:
            # Full analysis: Use mean_uncertainty_overall (all steps), consistent with TRACER
            mean_uncertainty = sim.summary.get('mean_uncertainty_overall')
            if mean_uncertainty is None:
                continue
        
        # Label encoding: Failure=1, Success=0
        ground_truth = 0 if sim.ground_truth_pass else 1
        
        y_scores_entropy.append(mean_uncertainty)
        y_true_entropy.append(ground_truth)
    
    # Calculate metrics for normalized entropy baseline
    if len(y_scores_entropy) >= 2 and len(np.unique(y_true_entropy)) == 2:
        try:
            y_scores_entropy = np.array(y_scores_entropy)
            y_true_entropy = np.array(y_true_entropy)
            
            # Calculate AUROC
            auroc = roc_auc_score(y_true_entropy, y_scores_entropy)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true_entropy, y_scores_entropy)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred = (y_scores_entropy >= optimal_threshold).astype(int)
            
            accuracy = accuracy_score(y_true_entropy, y_pred)
            precision = precision_score(y_true_entropy, y_pred, zero_division=0)
            recall = recall_score(y_true_entropy, y_pred, zero_division=0)
            
            # F1 score
            if precision + recall > 0:
                f1_score_val = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score_val = 0.0
            
            # Statistics by class
            failures = y_scores_entropy[y_true_entropy == 1]
            successes = y_scores_entropy[y_true_entropy == 0]
            
            results['normalized_entropy'] = {
                'auroc': float(auroc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score_val),
                'optimal_threshold': float(optimal_threshold),
                'num_samples': len(y_scores_entropy),
                'num_failures': int(np.sum(y_true_entropy == 1)),
                'num_successes': int(np.sum(y_true_entropy == 0)),
                'mean_tracer_failures': float(np.mean(failures)) if len(failures) > 0 else 0.0,
                'mean_tracer_successes': float(np.mean(successes)) if len(successes) > 0 else 0.0,
                'std_tracer_failures': float(np.std(failures)) if len(failures) > 0 else 0.0,
                'std_tracer_successes': float(np.std(successes)) if len(successes) > 0 else 0.0,
            }
            logger.info(f"Normalized Entropy baseline AUROC: {auroc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUROC for normalized entropy baseline: {e}")
    
    #########################################################################################
    # Baseline 2: Self-assessed Confidence (extract from original simulation data)
    #########################################################################################
    y_scores_confidence = []
    y_true_confidence = []
    
    if original_results is not None:
        for sim in original_results.simulations:
            # Get ground truth
            ground_truth_reward = sim.reward_info.reward if sim.reward_info else None
            if ground_truth_reward is None:
                continue
            
            ground_truth_pass = ground_truth_reward == 1.0
            ground_truth = 0 if ground_truth_pass else 1
            
            # Extract self-assessed confidence from messages based on role_filter
            confidences = []
            for msg in sim.messages:
                # Apply role filter
                if role_filter is not None:
                    if msg.role != role_filter:
                        continue
                else:
                    # Full analysis: Use ALL messages (both assistant and user)
                    # Only filter out system messages
                    if msg.role not in ["assistant", "user"]:
                        continue
                
                if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                    confidence = msg.uncertainty.get('self_assessed_confidence')
                    if confidence is not None:
                        confidences.append(confidence)
            
            if not confidences:
                continue
            
            mean_confidence = float(np.mean(confidences))
            y_scores_confidence.append(mean_confidence)
            y_true_confidence.append(ground_truth)
    
    # Calculate metrics for self-assessed confidence baseline
    if len(y_scores_confidence) >= 2 and len(np.unique(y_true_confidence)) == 2:
        try:
            y_scores_confidence = np.array(y_scores_confidence)
            y_true_confidence = np.array(y_true_confidence)
            
            # Calculate AUROC
            auroc = roc_auc_score(y_true_confidence, y_scores_confidence)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true_confidence, y_scores_confidence)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred = (y_scores_confidence >= optimal_threshold).astype(int)
            
            accuracy = accuracy_score(y_true_confidence, y_pred)
            precision = precision_score(y_true_confidence, y_pred, zero_division=0)
            recall = recall_score(y_true_confidence, y_pred, zero_division=0)
            
            # F1 score
            if precision + recall > 0:
                f1_score_val = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score_val = 0.0
            
            # Statistics by class
            failures = y_scores_confidence[y_true_confidence == 1]
            successes = y_scores_confidence[y_true_confidence == 0]
            
            results['self_assessed_confidence'] = {
                'auroc': float(auroc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score_val),
                'optimal_threshold': float(optimal_threshold),
                'num_samples': len(y_scores_confidence),
                'num_failures': int(np.sum(y_true_confidence == 1)),
                'num_successes': int(np.sum(y_true_confidence == 0)),
                'mean_tracer_failures': float(np.mean(failures)) if len(failures) > 0 else 0.0,
                'mean_tracer_successes': float(np.mean(successes)) if len(successes) > 0 else 0.0,
                'std_tracer_failures': float(np.std(failures)) if len(failures) > 0 else 0.0,
                'std_tracer_successes': float(np.std(successes)) if len(successes) > 0 else 0.0,
            }
            logger.info(f"Self-assessed Confidence baseline AUROC: {auroc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUROC for self-assessed confidence baseline: {e}")
    else:
        logger.info("Self-assessed confidence data not available or insufficient for baseline calculation")
    
    #########################################################################################
    # Baseline 3: SAUP (multiplicative weighting with RMS aggregation)
    #########################################################################################
    y_scores_saup = []
    y_true_saup = []
    
    for sim in analyzed_sims:
        if sim.ground_truth_pass is None:
            continue
        
        # Filter by role if specified
        if role_filter is not None:
            target_actor = "agent" if role_filter == "assistant" else "user"
            filtered_scores = [s for s in sim.uncertainty_scores if s.actor == target_actor]
        else:
            # Full analysis: Use ALL steps (both agent and user), consistent with TRACER
            filtered_scores = sim.uncertainty_scores
        
        if not filtered_scores:
            continue
        
        # Build step_data from filtered uncertainty_scores
        step_data = []
        for score in filtered_scores:
            # Extract do_agent and do_user based on do_type
            do_agent = None
            do_user = None
            if score.do_score is not None and score.do_type is not None:
                if score.do_type == 'agent_coherence':
                    do_agent = score.do_score
                elif score.do_type == 'user_coherence':
                    do_user = score.do_score
            
            step_data.append({
                'ui': score.normentropy_filtered_score,
                'da': score.da_score,
                'do_agent': do_agent,
                'do_user': do_user
            })
        
        if not step_data:
            continue
        
        # Calculate SAUP score
        saup_result = calculate_saup_score(step_data, config)
        saup_score = saup_result['saup_score']
        
        # Label encoding: Failure=1, Success=0
        ground_truth = 0 if sim.ground_truth_pass else 1
        
        y_scores_saup.append(saup_score)
        y_true_saup.append(ground_truth)
    
    # Calculate metrics for SAUP baseline
    if len(y_scores_saup) >= 2 and len(np.unique(y_true_saup)) == 2:
        try:
            y_scores_saup = np.array(y_scores_saup)
            y_true_saup = np.array(y_true_saup)
            
            # Calculate AUROC
            auroc = roc_auc_score(y_true_saup, y_scores_saup)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true_saup, y_scores_saup)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred = (y_scores_saup >= optimal_threshold).astype(int)
            
            accuracy = accuracy_score(y_true_saup, y_pred)
            precision = precision_score(y_true_saup, y_pred, zero_division=0)
            recall = recall_score(y_true_saup, y_pred, zero_division=0)
            
            # F1 score
            if precision + recall > 0:
                f1_score_val = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score_val = 0.0
            
            # Statistics by class
            failures = y_scores_saup[y_true_saup == 1]
            successes = y_scores_saup[y_true_saup == 0]
            
            results['saup'] = {
                'auroc': float(auroc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score_val),
                'optimal_threshold': float(optimal_threshold),
                'num_samples': len(y_scores_saup),
                'num_failures': int(np.sum(y_true_saup == 1)),
                'num_successes': int(np.sum(y_true_saup == 0)),
                'mean_tracer_failures': float(np.mean(failures)) if len(failures) > 0 else 0.0,
                'mean_tracer_successes': float(np.mean(successes)) if len(successes) > 0 else 0.0,
                'std_tracer_failures': float(np.std(failures)) if len(failures) > 0 else 0.0,
                'std_tracer_successes': float(np.std(successes)) if len(successes) > 0 else 0.0,
            }
            logger.info(f"SAUP baseline AUROC: {auroc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUROC for SAUP baseline: {e}")
    
    if not results:
        logger.warning("No baseline AUROCs could be calculated")
        return None
    
    logger.info(f"Calculated {len(results)} baseline metric(s)")
    return results


def calculate_auarc_metrics(
    analyzed_sims: list[SimulationUncertainty]
) -> Optional[AUARCMetrics]:
    """
    Calculate AUARC (Area Under Accuracy-Rejection Curve) metrics for selective prediction.
    
    AUARC measures how well the uncertainty score (TRACER) can identify which predictions
    to reject in order to improve accuracy. This is a key metric for selective prediction
    systems where the agent can choose to abstain from making predictions on uncertain cases.
    
    The algorithm:
    1. Sort all samples by TRACER score in descending order (highest uncertainty first)
    2. For each coverage level (1.0, 0.95, ..., 0.05):
       - Keep only the most confident samples (lowest TRACER scores)
       - Calculate accuracy on the retained samples
    3. Compute the area under the accuracy-coverage curve using trapezoidal rule
    
    Args:
        analyzed_sims: List of analyzed simulations with TRACER metrics and ground truth
        
    Returns:
        AUARCMetrics if calculation successful, None otherwise
    
    Example:
        - If rejecting top 20% uncertain samples improves accuracy from 60% to 75%,
          this indicates TRACER is successfully identifying problematic cases
        - AUARC summarizes this behavior across all coverage levels
    """
    # Extract TRACER scores and ground truth labels
    samples = []
    
    for sim in analyzed_sims:
        # Need both ground truth and TRACER score
        if sim.ground_truth_pass is None or sim.tracer_metrics is None:
            continue
        
        tracer_score = sim.tracer_metrics.get('tracer_score')
        if tracer_score is None:
            continue
        
        # Store (TRACER score, is_correct)
        is_correct = sim.ground_truth_pass
        samples.append((tracer_score, is_correct))
    
    # Need at least 10 samples for meaningful AUARC
    if len(samples) < 10:
        logger.warning(f"Not enough samples for AUARC calculation ({len(samples)} < 10). Skipping.")
        return None
    
    try:
        # Sort by TRACER score descending (most uncertain first)
        samples_sorted = sorted(samples, key=lambda x: x[0], reverse=True)
        
        # Extract sorted arrays
        tracer_scores = np.array([s[0] for s in samples_sorted])
        correctness = np.array([s[1] for s in samples_sorted])
        
        total_samples = len(samples)
        num_successes = int(np.sum(correctness))
        num_failures = total_samples - num_successes
        
        # Calculate baseline accuracy (no rejection)
        baseline_accuracy = float(np.mean(correctness))
        
        # Calculate accuracy at different coverage levels
        # Coverage = fraction of samples we keep (1.0 = keep all, 0.5 = keep half)
        # We reject from the high-uncertainty end
        
        coverages = []
        accuracies = []
        
        # Generate coverage points: 100%, 95%, 90%, ..., 5%
        coverage_points = np.linspace(1.0, 0.05, 20)
        
        for coverage in coverage_points:
            # Number of samples to keep
            n_keep = int(np.ceil(coverage * total_samples))
            n_keep = max(1, min(n_keep, total_samples))  # Clamp to valid range
            
            # Reject the most uncertain samples (first n_reject samples)
            # Keep the most confident samples (last n_keep samples)
            n_reject = total_samples - n_keep
            kept_samples = correctness[n_reject:]
            
            # Calculate accuracy on kept samples
            if len(kept_samples) > 0:
                accuracy = float(np.mean(kept_samples))
            else:
                accuracy = 0.0
            
            coverages.append(float(coverage))
            accuracies.append(accuracy)
        
        # Compute AUARC using trapezoidal rule
        # AUARC = integral of accuracy over coverage [0, 1]
        # Need to reverse arrays since coverage goes from 1.0 to 0.05 (decreasing)
        # Trapezoidal rule requires increasing x values
        auarc = float(np.trapezoid(list(reversed(accuracies)), list(reversed(coverages))))
        
        # Find maximum achievable accuracy
        max_accuracy = float(np.max(accuracies))
        
        # Find accuracy at specific coverage levels
        accuracy_at_50 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.5))]
        accuracy_at_80 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.8))]
        
        # Find optimal coverage (maximizes accuracy - baseline_accuracy)
        improvements = np.array(accuracies) - baseline_accuracy
        optimal_idx = np.argmax(improvements)
        optimal_coverage = float(coverages[optimal_idx])
        optimal_accuracy = float(accuracies[optimal_idx])
        
        logger.info(f"AUARC: {auarc:.4f}, Max Accuracy: {max_accuracy:.4f}, Baseline: {baseline_accuracy:.4f}")
        
        return AUARCMetrics(
            auarc=auarc,
            max_accuracy=max_accuracy,
            accuracy_at_50_coverage=float(accuracy_at_50),
            accuracy_at_80_coverage=float(accuracy_at_80),
            optimal_coverage=optimal_coverage,
            optimal_accuracy=optimal_accuracy,
            baseline_accuracy=baseline_accuracy,
            num_samples=total_samples,
            num_failures=num_failures,
            num_successes=num_successes,
            coverages=coverages,
            accuracies=accuracies
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate AUARC: {e}")
        return None


def calculate_early_warning_metrics(
    analyzed_sims: list[SimulationUncertainty],
    baseline_aurocs: Optional[dict] = None,
    tracer_auroc: Optional[AUROCMetrics] = None
) -> Optional[dict[str, EarlyWarningMetrics]]:
    """
    Calculate early warning metrics using threshold crossing detection.
    
    This analysis shows when each metric first crosses its optimal failure threshold,
    indicating early detection of problems.
    
    The algorithm:
    1. Extract optimal thresholds from AUROC analysis for each metric
    2. For each failed simulation and each metric:
       - Find the FIRST turn where the metric crosses its threshold
       - Calculate the percentage of trajectory where this first crossing occurs
    3. Aggregate statistics across all failed simulations
    
    Args:
        analyzed_sims: List of analyzed simulations with per-turn metrics
        baseline_aurocs: Dictionary of baseline AUROC metrics (contains optimal thresholds)
        tracer_auroc: TRACER AUROC metrics (contains optimal threshold)
        
    Returns:
        Dictionary mapping metric name to EarlyWarningMetrics, or None if insufficient data
    """
    # Filter only failed simulations
    failed_sims = [sim for sim in analyzed_sims if sim.ground_truth_pass is False]
    
    if len(failed_sims) < 5:
        logger.warning(f"Not enough failed simulations for early warning analysis ({len(failed_sims)} < 5). Skipping.")
        return None
    
    logger.info(f"Analyzing threshold crossing for early warning across {len(failed_sims)} failed simulations")
    
    results = {}
    
    # Helper function to find first threshold crossing and calculate percentage
    def find_first_crossing_percentage(values: list[float], threshold: float, total_turns: int) -> Optional[float]:
        """Find the first turn where value crosses threshold and return as percentage of trajectory."""
        if not values or total_turns == 0:
            return None
        
        # Find first turn where value exceeds threshold
        for turn_idx, value in enumerate(values):
            if value > threshold:
                # Calculate percentage of trajectory (0-100)
                percentage = (turn_idx / (total_turns - 1)) * 100 if total_turns > 1 else 0.0
                return float(percentage)
        
        # If threshold never crossed, return 100% (signals at the very end)
        return 100.0
    
    #########################################################################################
    # Metric 1: Normalized Entropy (per-turn normentropy_score)
    #########################################################################################
    if baseline_aurocs and 'normalized_entropy' in baseline_aurocs:
        threshold_entropy = baseline_aurocs['normalized_entropy']['optimal_threshold']
        logger.info(f"Using Normalized Entropy threshold: {threshold_entropy:.4f}")
        
        entropy_crossings = []
        
        for sim in failed_sims:
            # Extract per-turn normalized entropy
            normentropy_scores = [turn.normentropy_score for turn in sim.uncertainty_scores if turn.normentropy_score is not None]
            total_turns = len(normentropy_scores)
            
            if total_turns > 0:
                crossing_pct = find_first_crossing_percentage(normentropy_scores, threshold_entropy, total_turns)
                if crossing_pct is not None:
                    entropy_crossings.append(crossing_pct)
        
        if entropy_crossings:
            results['normalized_entropy'] = EarlyWarningMetrics(
                num_failed_tasks=len(entropy_crossings),
                metric_name='Normalized Entropy',
                mean_peak_percentage=float(np.mean(entropy_crossings)),
                median_peak_percentage=float(np.median(entropy_crossings)),
                std_peak_percentage=float(np.std(entropy_crossings)),
                percent_before_10=float(np.mean([p < 10 for p in entropy_crossings]) * 100),
                percent_before_20=float(np.mean([p < 20 for p in entropy_crossings]) * 100),
                percent_before_30=float(np.mean([p < 30 for p in entropy_crossings]) * 100),
                percent_before_40=float(np.mean([p < 40 for p in entropy_crossings]) * 100),
                percent_before_50=float(np.mean([p < 50 for p in entropy_crossings]) * 100),
                percent_before_60=float(np.mean([p < 60 for p in entropy_crossings]) * 100),
                percent_before_70=float(np.mean([p < 70 for p in entropy_crossings]) * 100),
                percent_before_80=float(np.mean([p < 80 for p in entropy_crossings]) * 100),
                percent_before_90=float(np.mean([p < 90 for p in entropy_crossings]) * 100)
            )
            logger.info(f"Normalized Entropy: mean first crossing at {results['normalized_entropy'].mean_peak_percentage:.1f}% of trajectory")
    else:
        logger.warning("Normalized Entropy AUROC not available, skipping early warning analysis")
    
    #########################################################################################
    # Metric 2: Self-Assessed Confidence (from per-turn selfconf_score)
    # NOTE: selfconf_score is CONFIDENCE (higher = more confident)
    # We convert to UNCERTAINTY by using (1 - selfconf_score) for consistency with other metrics
    #########################################################################################
    if baseline_aurocs and 'self_assessed_confidence' in baseline_aurocs:
        # Threshold is for uncertainty (1 - confidence), already computed in AUROC
        threshold_confidence = baseline_aurocs['self_assessed_confidence']['optimal_threshold']
        logger.info(f"Using Self-Assessed Confidence (as uncertainty) threshold: {threshold_confidence:.4f}")
        
        confidence_crossings = []
        
        for sim in failed_sims:
            # Extract per-turn self-assessed confidence and convert to uncertainty
            # selfconf_score is confidence (0-1), so (1 - selfconf_score) is uncertainty
            uncertainties = [turn.selfconf_score for turn in sim.uncertainty_scores if turn.selfconf_score is not None]
            total_turns = len(uncertainties)
            
            if total_turns > 0:
                crossing_pct = find_first_crossing_percentage(uncertainties, threshold_confidence, total_turns)
                if crossing_pct is not None:
                    confidence_crossings.append(crossing_pct)
        
        if confidence_crossings:
            results['self_assessed_confidence'] = EarlyWarningMetrics(
                num_failed_tasks=len(confidence_crossings),
                metric_name='Self-Assessed Confidence',
                mean_peak_percentage=float(np.mean(confidence_crossings)),
                median_peak_percentage=float(np.median(confidence_crossings)),
                std_peak_percentage=float(np.std(confidence_crossings)),
                percent_before_10=float(np.mean([p < 10 for p in confidence_crossings]) * 100),
                percent_before_20=float(np.mean([p < 20 for p in confidence_crossings]) * 100),
                percent_before_30=float(np.mean([p < 30 for p in confidence_crossings]) * 100),
                percent_before_40=float(np.mean([p < 40 for p in confidence_crossings]) * 100),
                percent_before_50=float(np.mean([p < 50 for p in confidence_crossings]) * 100),
                percent_before_60=float(np.mean([p < 60 for p in confidence_crossings]) * 100),
                percent_before_70=float(np.mean([p < 70 for p in confidence_crossings]) * 100),
                percent_before_80=float(np.mean([p < 80 for p in confidence_crossings]) * 100),
                percent_before_90=float(np.mean([p < 90 for p in confidence_crossings]) * 100)
            )
            logger.info(f"Self-Assessed Confidence: mean first crossing at {results['self_assessed_confidence'].mean_peak_percentage:.1f}% of trajectory")
    else:
        logger.warning("Self-Assessed Confidence AUROC not available, skipping early warning analysis")
    
    #########################################################################################
    # Metric 3: SAUP (cumulative per-turn)
    #########################################################################################
    if baseline_aurocs and 'saup' in baseline_aurocs:
        threshold_saup = baseline_aurocs['saup']['optimal_threshold']
        logger.info(f"Using SAUP threshold: {threshold_saup:.4f}")
        
        saup_crossings = []
        
        for sim in failed_sims:
            # Extract cumulative SAUP scores
            saup_scores = [turn.saup_score for turn in sim.uncertainty_scores if turn.saup_score is not None]
            total_turns = len(saup_scores)
            
            if total_turns > 0:
                crossing_pct = find_first_crossing_percentage(saup_scores, threshold_saup, total_turns)
                if crossing_pct is not None:
                    saup_crossings.append(crossing_pct)
        
        if saup_crossings:
            results['saup'] = EarlyWarningMetrics(
                num_failed_tasks=len(saup_crossings),
                metric_name='SAUP',
                mean_peak_percentage=float(np.mean(saup_crossings)),
                median_peak_percentage=float(np.median(saup_crossings)),
                std_peak_percentage=float(np.std(saup_crossings)),
                percent_before_10=float(np.mean([p < 10 for p in saup_crossings]) * 100),
                percent_before_20=float(np.mean([p < 20 for p in saup_crossings]) * 100),
                percent_before_30=float(np.mean([p < 30 for p in saup_crossings]) * 100),
                percent_before_40=float(np.mean([p < 40 for p in saup_crossings]) * 100),
                percent_before_50=float(np.mean([p < 50 for p in saup_crossings]) * 100),
                percent_before_60=float(np.mean([p < 60 for p in saup_crossings]) * 100),
                percent_before_70=float(np.mean([p < 70 for p in saup_crossings]) * 100),
                percent_before_80=float(np.mean([p < 80 for p in saup_crossings]) * 100),
                percent_before_90=float(np.mean([p < 90 for p in saup_crossings]) * 100)
            )
            logger.info(f"SAUP: mean first crossing at {results['saup'].mean_peak_percentage:.1f}% of trajectory")
    else:
        logger.warning("SAUP AUROC not available, skipping early warning analysis")
    
    #########################################################################################
    # Metric 4: TRACER (cumulative per-turn)
    #########################################################################################
    if tracer_auroc is not None:
        threshold_tracer = tracer_auroc.optimal_threshold
        logger.info(f"Using TRACER threshold: {threshold_tracer:.4f}")
        
        tracer_crossings = []
        
        for sim in failed_sims:
            # Extract cumulative TRACER scores
            tracer_scores = [turn.tracer_score for turn in sim.uncertainty_scores if turn.tracer_score is not None]
            total_turns = len(tracer_scores)
            
            if total_turns > 0:
                crossing_pct = find_first_crossing_percentage(tracer_scores, threshold_tracer, total_turns)
                if crossing_pct is not None:
                    tracer_crossings.append(crossing_pct)
        
        if tracer_crossings:
            results['tracer'] = EarlyWarningMetrics(
                num_failed_tasks=len(tracer_crossings),
                metric_name='TRACER',
                mean_peak_percentage=float(np.mean(tracer_crossings)),
                median_peak_percentage=float(np.median(tracer_crossings)),
                std_peak_percentage=float(np.std(tracer_crossings)),
                percent_before_10=float(np.mean([p < 10 for p in tracer_crossings]) * 100),
                percent_before_20=float(np.mean([p < 20 for p in tracer_crossings]) * 100),
                percent_before_30=float(np.mean([p < 30 for p in tracer_crossings]) * 100),
                percent_before_40=float(np.mean([p < 40 for p in tracer_crossings]) * 100),
                percent_before_50=float(np.mean([p < 50 for p in tracer_crossings]) * 100),
                percent_before_60=float(np.mean([p < 60 for p in tracer_crossings]) * 100),
                percent_before_70=float(np.mean([p < 70 for p in tracer_crossings]) * 100),
                percent_before_80=float(np.mean([p < 80 for p in tracer_crossings]) * 100),
                percent_before_90=float(np.mean([p < 90 for p in tracer_crossings]) * 100)
            )
            logger.info(f"TRACER: mean first crossing at {results['tracer'].mean_peak_percentage:.1f}% of trajectory")
    else:
        logger.warning("TRACER AUROC not available, skipping early warning analysis")
    
    if not results:
        logger.warning("No early warning metrics could be calculated")
        return None
    
    logger.info(f"Calculated early warning metrics for {len(results)} metric(s)")
    return results


def calculate_baseline_auarcs(
    analyzed_sims: list[SimulationUncertainty],
    original_results: Results = None,
    config: Optional[TRACERConfig] = None
) -> Optional[dict]:
    """
    Calculate AUARC metrics for baseline predictors (selective prediction evaluation).
    
    This function evaluates baseline metrics to compare against TRACER:
    1. Normalized Entropy only (mean U_i)
    2. Self-assessed Confidence (extracted from original simulation messages if available)
    3. SAUP (multiplicative weighting with RMS aggregation)
    
    Args:
        analyzed_sims: List of analyzed simulations with summary statistics
        original_results: Original Results object to extract self-assessed confidence
        config: TRACER configuration (only needed for SAUP)
        
    Returns:
        Dictionary with baseline AUARC metrics matching TRACER schema, or None if calculation fails
    """
    if config is None:
        config = TRACERConfig()
    
    results = {}
    
    #########################################################################################
    # Baseline 1: Normalized Entropy (U_i) only
    #########################################################################################
    samples_entropy = []
    
    for sim in analyzed_sims:
        if sim.ground_truth_pass is None:
            continue
        
        # Use mean_uncertainty_overall (all steps), consistent with TRACER
        mean_uncertainty = sim.summary.get('mean_uncertainty_overall')
        if mean_uncertainty is None:
            continue
        
        # Store (uncertainty_score, is_correct)
        is_correct = sim.ground_truth_pass
        samples_entropy.append((mean_uncertainty, is_correct))
    
    # Calculate AUARC for normalized entropy baseline
    if len(samples_entropy) >= 10:
        try:
            # Sort by uncertainty descending (most uncertain first)
            samples_sorted = sorted(samples_entropy, key=lambda x: x[0], reverse=True)
            
            tracer_scores = np.array([s[0] for s in samples_sorted])
            correctness = np.array([s[1] for s in samples_sorted])
            
            total_samples = len(samples_entropy)
            num_successes = int(np.sum(correctness))
            num_failures = total_samples - num_successes
            baseline_accuracy = float(np.mean(correctness))
            
            # Calculate accuracy at different coverage levels
            coverages = []
            accuracies = []
            coverage_points = np.linspace(1.0, 0.05, 20)
            
            for coverage in coverage_points:
                n_keep = int(np.ceil(coverage * total_samples))
                n_keep = max(1, min(n_keep, total_samples))
                n_reject = total_samples - n_keep
                kept_samples = correctness[n_reject:]
                
                if len(kept_samples) > 0:
                    accuracy = float(np.mean(kept_samples))
                else:
                    accuracy = 0.0
                
                coverages.append(float(coverage))
                accuracies.append(accuracy)
            
            # Compute AUARC
            # Need to reverse arrays since coverage goes from 1.0 to 0.05 (decreasing)
            auarc = float(np.trapezoid(list(reversed(accuracies)), list(reversed(coverages))))
            max_accuracy = float(np.max(accuracies))
            accuracy_at_50 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.5))]
            accuracy_at_80 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.8))]
            
            improvements = np.array(accuracies) - baseline_accuracy
            optimal_idx = np.argmax(improvements)
            optimal_coverage = float(coverages[optimal_idx])
            optimal_accuracy = float(accuracies[optimal_idx])
            
            results['normalized_entropy'] = {
                'auarc': auarc,
                'max_accuracy': max_accuracy,
                'accuracy_at_50_coverage': float(accuracy_at_50),
                'accuracy_at_80_coverage': float(accuracy_at_80),
                'optimal_coverage': optimal_coverage,
                'optimal_accuracy': optimal_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'num_samples': total_samples,
                'num_failures': num_failures,
                'num_successes': num_successes,
                'coverages': coverages,
                'accuracies': accuracies
            }
            logger.info(f"Normalized Entropy baseline AUARC: {auarc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUARC for normalized entropy baseline: {e}")
    
    #########################################################################################
    # Baseline 2: Self-assessed Confidence (extract from original simulation data)
    #########################################################################################
    samples_confidence = []
    
    if original_results is not None:
        for sim in original_results.simulations:
            # Get ground truth
            ground_truth_reward = sim.reward_info.reward if sim.reward_info else None
            if ground_truth_reward is None:
                continue
            
            ground_truth_pass = ground_truth_reward == 1.0
            is_correct = ground_truth_pass
            
            # Extract self-assessed confidence from messages (all roles)
            confidences = []
            for msg in sim.messages:
                if msg.role not in ["assistant", "user"]:
                    continue
                
                if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                    confidence = msg.uncertainty.get('self_assessed_confidence')
                    if confidence is not None:
                        confidences.append(confidence)
            
            if not confidences:
                continue
            
            mean_confidence = float(np.mean(confidences))
            samples_confidence.append((mean_confidence, is_correct))
    
    # Calculate AUARC for self-assessed confidence baseline
    if len(samples_confidence) >= 10:
        try:
            # Sort by confidence descending (most uncertain/low confidence first)
            # Higher confidence = lower uncertainty, so reverse sort
            samples_sorted = sorted(samples_confidence, key=lambda x: x[0], reverse=True)
            
            confidence_scores = np.array([s[0] for s in samples_sorted])
            correctness = np.array([s[1] for s in samples_sorted])
            
            total_samples = len(samples_confidence)
            num_successes = int(np.sum(correctness))
            num_failures = total_samples - num_successes
            baseline_accuracy = float(np.mean(correctness))
            
            # Calculate accuracy at different coverage levels
            coverages = []
            accuracies = []
            coverage_points = np.linspace(1.0, 0.05, 20)
            
            for coverage in coverage_points:
                n_keep = int(np.ceil(coverage * total_samples))
                n_keep = max(1, min(n_keep, total_samples))
                n_reject = total_samples - n_keep
                kept_samples = correctness[n_reject:]
                
                if len(kept_samples) > 0:
                    accuracy = float(np.mean(kept_samples))
                else:
                    accuracy = 0.0
                
                coverages.append(float(coverage))
                accuracies.append(accuracy)
            
            # Compute AUARC
            # Need to reverse arrays since coverage goes from 1.0 to 0.05 (decreasing)
            auarc = float(np.trapezoid(list(reversed(accuracies)), list(reversed(coverages))))
            max_accuracy = float(np.max(accuracies))
            accuracy_at_50 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.5))]
            accuracy_at_80 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.8))]
            
            improvements = np.array(accuracies) - baseline_accuracy
            optimal_idx = np.argmax(improvements)
            optimal_coverage = float(coverages[optimal_idx])
            optimal_accuracy = float(accuracies[optimal_idx])
            
            results['self_assessed_confidence'] = {
                'auarc': auarc,
                'max_accuracy': max_accuracy,
                'accuracy_at_50_coverage': float(accuracy_at_50),
                'accuracy_at_80_coverage': float(accuracy_at_80),
                'optimal_coverage': optimal_coverage,
                'optimal_accuracy': optimal_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'num_samples': total_samples,
                'num_failures': num_failures,
                'num_successes': num_successes,
                'coverages': coverages,
                'accuracies': accuracies
            }
            logger.info(f"Self-assessed Confidence baseline AUARC: {auarc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUARC for self-assessed confidence baseline: {e}")
    else:
        logger.info("Self-assessed confidence data not available or insufficient for baseline AUARC calculation")
    
    #########################################################################################
    # Baseline 3: SAUP (multiplicative weighting with RMS aggregation)
    #########################################################################################
    samples_saup = []
    
    for sim in analyzed_sims:
        if sim.ground_truth_pass is None:
            continue
        
        # Use ALL steps (both agent and user), consistent with TRACER
        filtered_scores = sim.uncertainty_scores
        
        if not filtered_scores:
            continue
        
        # Build step_data from uncertainty_scores
        step_data = []
        for score in filtered_scores:
            # Extract do_agent and do_user based on do_type
            do_agent = None
            do_user = None
            if score.do_score is not None and score.do_type is not None:
                if score.do_type == 'agent_coherence':
                    do_agent = score.do_score
                elif score.do_type == 'user_coherence':
                    do_user = score.do_score
            
            step_data.append({
                'ui': score.normentropy_filtered_score,
                'da': score.da_score,
                'do_agent': do_agent,
                'do_user': do_user
            })
        
        if not step_data:
            continue
        
        # Calculate SAUP score
        saup_result = calculate_saup_score(step_data, config)
        saup_score = saup_result['saup_score']
        
        # Store (saup_score, is_correct)
        is_correct = sim.ground_truth_pass
        samples_saup.append((saup_score, is_correct))
    
    # Calculate AUARC for SAUP baseline
    if len(samples_saup) >= 10:
        try:
            # Sort by SAUP descending (most uncertain first)
            samples_sorted = sorted(samples_saup, key=lambda x: x[0], reverse=True)
            
            saup_scores = np.array([s[0] for s in samples_sorted])
            correctness = np.array([s[1] for s in samples_sorted])
            
            total_samples = len(samples_saup)
            num_successes = int(np.sum(correctness))
            num_failures = total_samples - num_successes
            baseline_accuracy = float(np.mean(correctness))
            
            # Calculate accuracy at different coverage levels
            coverages = []
            accuracies = []
            coverage_points = np.linspace(1.0, 0.05, 20)
            
            for coverage in coverage_points:
                n_keep = int(np.ceil(coverage * total_samples))
                n_keep = max(1, min(n_keep, total_samples))
                n_reject = total_samples - n_keep
                kept_samples = correctness[n_reject:]
                
                if len(kept_samples) > 0:
                    accuracy = float(np.mean(kept_samples))
                else:
                    accuracy = 0.0
                
                coverages.append(float(coverage))
                accuracies.append(accuracy)
            
            # Compute AUARC
            # Need to reverse arrays since coverage goes from 1.0 to 0.05 (decreasing)
            auarc = float(np.trapezoid(list(reversed(accuracies)), list(reversed(coverages))))
            max_accuracy = float(np.max(accuracies))
            accuracy_at_50 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.5))]
            accuracy_at_80 = accuracies[np.argmin(np.abs(np.array(coverages) - 0.8))]
            
            improvements = np.array(accuracies) - baseline_accuracy
            optimal_idx = np.argmax(improvements)
            optimal_coverage = float(coverages[optimal_idx])
            optimal_accuracy = float(accuracies[optimal_idx])
            
            results['saup'] = {
                'auarc': auarc,
                'max_accuracy': max_accuracy,
                'accuracy_at_50_coverage': float(accuracy_at_50),
                'accuracy_at_80_coverage': float(accuracy_at_80),
                'optimal_coverage': optimal_coverage,
                'optimal_accuracy': optimal_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'num_samples': total_samples,
                'num_failures': num_failures,
                'num_successes': num_successes,
                'coverages': coverages,
                'accuracies': accuracies
            }
            logger.info(f"SAUP baseline AUARC: {auarc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to calculate AUARC for SAUP baseline: {e}")
    
    if not results:
        logger.warning("No baseline AUARCs could be calculated")
        return None
    
    logger.info(f"Calculated AUARC for {len(results)} baseline metric(s)")
    return results


def analyze_results(results: Results, config: TRACERConfig, verbose: bool = False, calculate_auroc: bool = True) -> UncertaintyAnalysis:
    """
    Analyze all simulations in the results.

    Args:
        results: Tau-2 simulation results
        config: TRACER configuration for weighted aggregation
        verbose: If True, include detailed statistics
        calculate_auroc: If True, calculate AUROC metrics for failure prediction

    Returns:
        UncertaintyAnalysis: Complete analysis results
    """
    analyzed_sims = []

    for simulation in results.simulations:
        sim_dict = simulation.model_dump()
        analyzed = analyze_simulation(sim_dict, config, verbose=verbose)
        analyzed_sims.append(analyzed)

    # Create metadata
    metadata = {
        "source_timestamp": results.timestamp or "unknown",
        "total_simulations": len(analyzed_sims),
        "domain": results.info.environment_info.domain_name,
        "agent_llm": results.info.agent_info.llm,
        "user_llm": results.info.user_info.llm,
    }
    
    # Calculate AUROC metrics if requested and TRACER scores are available
    auroc_metrics = None
    auarc_metrics = None
    baseline_aurocs = None
    baseline_auarcs = None
    ablation_studies = None
    early_warning_metrics = None
    
    if calculate_auroc:
        # Full analysis (default - uses all turns for TRACER, agent-only for baselines)
        logger.info("Calculating AUROC metrics for failure prediction...")
        auroc_metrics = calculate_auroc_metrics(analyzed_sims)
        baseline_aurocs = calculate_baseline_aurocs(analyzed_sims, original_results=results)
        
        # Calculate AUARC metrics (only for full analysis with both agent and user signals)
        logger.info("Calculating AUARC metrics for selective prediction...")
        auarc_metrics = calculate_auarc_metrics(analyzed_sims)
        baseline_auarcs = calculate_baseline_auarcs(analyzed_sims, original_results=results, config=config)
        
        # Calculate early warning metrics (requires AUROC thresholds)
        logger.info("Calculating early warning metrics...")
        early_warning_metrics = calculate_early_warning_metrics(
            analyzed_sims, 
            baseline_aurocs=baseline_aurocs,
            tracer_auroc=auroc_metrics
        )
        
        # Ablation studies: isolate assistant-only and user-only
        logger.info("Running ablation studies...")
        ablation_studies = {}
        
        # Ablation 1: Assistant-only (role="assistant", actor="agent")
        logger.info("  - Assistant-only ablation")
        assistant_auroc = calculate_auroc_metrics(
            analyzed_sims, 
            role_filter="assistant", 
            config=config
        )
        assistant_baselines = calculate_baseline_aurocs(
            analyzed_sims, 
            original_results=results,
            role_filter="assistant",
            config=config
        )
        
        if assistant_auroc or assistant_baselines:
            ablation_studies['assistant_only'] = {
                'description': 'Analysis using only assistant (agent) turns',
                'auroc_metrics': assistant_auroc,
                'baseline_aurocs': assistant_baselines
            }
        
        # Ablation 2: User-only (role="user", actor="user")
        logger.info("  - User-only ablation")
        user_auroc = calculate_auroc_metrics(
            analyzed_sims, 
            role_filter="user",
            config=config
        )
        user_baselines = calculate_baseline_aurocs(
            analyzed_sims, 
            original_results=results,
            role_filter="user",
            config=config
        )
        
        if user_auroc or user_baselines:
            ablation_studies['user_only'] = {
                'description': 'Analysis using only user turns',
                'auroc_metrics': user_auroc,
                'baseline_aurocs': user_baselines
            }
        
        logger.info(f"Completed {len(ablation_studies)} ablation studies")

    return UncertaintyAnalysis(
        metadata=metadata,
        results=analyzed_sims,
        auroc_metrics=auroc_metrics,
        auarc_metrics=auarc_metrics,
        baseline_aurocs=baseline_aurocs,
        baseline_auarcs=baseline_auarcs,
        ablation_studies=ablation_studies,
        early_warning_metrics=early_warning_metrics
    )


def print_uncertainty_summary_from_results(
    results: Results, console: Optional[Console] = None
) -> None:
    """
    Print uncertainty summary directly from simulation results.
    
    This is a lightweight version that extracts and displays uncertainty
    statistics from simulation results that have embedded uncertainty data
    (i.e., simulations run with --calculate-uncertainty).
    
    Args:
        results: Simulation results with embedded uncertainty data
        console: Rich console (creates one if not provided)
    """
    if console is None:
        console = Console()
    
    # Collect all uncertainty and semantic distance scores
    all_agent_uncertainties = []
    all_user_uncertainties = []
    all_da_scores = []
    all_do_scores = []
    all_do_agent = []
    all_do_user = []
    sims_with_uncertainty = 0
    
    for sim in results.simulations:
        has_uncertainty = False
        for msg in sim.messages:
            # Only process messages that have the uncertainty attribute (AssistantMessage, UserMessage)
            if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                has_uncertainty = True
                uncertainty_score = msg.uncertainty.get('normalized_entropy', 0.0)
                if msg.role == "assistant":
                    all_agent_uncertainties.append(uncertainty_score)
                elif msg.role == "user":
                    all_user_uncertainties.append(uncertainty_score)
            
            # Collect semantic distance metrics
            if hasattr(msg, 'da_score') and msg.da_score is not None:
                all_da_scores.append(msg.da_score)
            if hasattr(msg, 'do_score') and msg.do_score is not None:
                all_do_scores.append(msg.do_score)
                if hasattr(msg, 'do_type'):
                    if msg.do_type == "agent_coherence":
                        all_do_agent.append(msg.do_score)
                    elif msg.do_type == "user_coherence":
                        all_do_user.append(msg.do_score)
        if has_uncertainty:
            sims_with_uncertainty += 1
    
    # If no uncertainty data found, return early
    if not all_agent_uncertainties and not all_user_uncertainties:
        return
    
    # Print summary
    console.print("\n" + "=" * 80, style="dim")
    console.print(
        "UNCERTAINTY ANALYSIS SUMMARY", style="bold cyan", justify="center"
    )
    console.print("=" * 80 + "\n", style="dim")
    
    # Metadata
    console.print(f"[bold]Domain:[/bold] {results.info.environment_info.domain_name}")
    console.print(f"[bold]Agent Model:[/bold] {results.info.agent_info.llm}")
    console.print(f"[bold]User Model:[/bold] {results.info.user_info.llm}")
    console.print(f"[bold]Simulations with Uncertainty:[/bold] {sims_with_uncertainty}/{len(results.simulations)}\n")
    
    # Overall statistics
    console.print("[bold cyan]Overall Statistics[/bold cyan]")
    
    if all_agent_uncertainties:
        console.print("\n[bold]Agent Reasoning Uncertainty (U_i,agent):[/bold]")
        console.print(f"  Mean: {np.mean(all_agent_uncertainties):.4f}")
        console.print(f"  Std:  {np.std(all_agent_uncertainties):.4f}")
        console.print(f"  Min:  {np.min(all_agent_uncertainties):.4f}")
        console.print(f"  Max:  {np.max(all_agent_uncertainties):.4f}")
        console.print(f"  Total turns: {len(all_agent_uncertainties)}")
    
    if all_user_uncertainties:
        console.print("\n[bold]User Confusion (U_i,user):[/bold]")
        console.print(f"  Mean: {np.mean(all_user_uncertainties):.4f}")
        console.print(f"  Std:  {np.std(all_user_uncertainties):.4f}")
        console.print(f"  Min:  {np.min(all_user_uncertainties):.4f}")
        console.print(f"  Max:  {np.max(all_user_uncertainties):.4f}")
        console.print(f"  Total turns: {len(all_user_uncertainties)}")
    
    # Semantic distance metrics
    if all_da_scores or all_do_scores:
        console.print("\n[bold cyan]Situational Awareness Metrics[/bold cyan]")
        
        if all_da_scores:
            console.print("\n[bold]Local Repetition (Looping Penalty):[/bold]")
            console.print(f"  Mean: {np.mean(all_da_scores):.4f}")
            console.print(f"  Std:  {np.std(all_da_scores):.4f}")
            console.print(f"  Min:  {np.min(all_da_scores):.4f}")
            console.print(f"  Max:  {np.max(all_da_scores):.4f}")
            console.print(f"  Total measurements: {len(all_da_scores)}")
            console.print(f"  [dim]Note: High values indicate agent is repeating itself (stuck in loops)[/dim]")
        
        if all_do_scores:
            console.print("\n[bold]Inference Gap (Do):[/bold]")
            console.print(f"  Mean: {np.mean(all_do_scores):.4f}")
            console.print(f"  Std:  {np.std(all_do_scores):.4f}")
            console.print(f"  Min:  {np.min(all_do_scores):.4f}")
            console.print(f"  Max:  {np.max(all_do_scores):.4f}")
            console.print(f"  Total measurements: {len(all_do_scores)}")
            
            if all_do_agent:
                console.print(f"\n  [dim]Agent Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_agent):.4f}")
                console.print(f"    Count: {len(all_do_agent)}")
            
            if all_do_user:
                console.print(f"\n  [dim]User Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_user):.4f}")
                console.print(f"    Count: {len(all_do_user)}")
    
    # TRACER Aggregation Scores
    all_tracer_scores = []
    tracer_passed = []
    tracer_failed = []
    
    for sim in results.simulations:
        if hasattr(sim, 'tracer_metrics') and sim.tracer_metrics is not None:
            tracer_score = sim.tracer_metrics.get('tracer_score')
            if tracer_score is not None:
                all_tracer_scores.append(tracer_score)
                
                # Track by ground truth
                if hasattr(sim, 'reward_info') and sim.reward_info is not None:
                    reward = sim.reward_info.reward
                    if reward == 1.0:
                        tracer_passed.append(tracer_score)
                    else:
                        tracer_failed.append(tracer_score)
    
    if all_tracer_scores:
        console.print("\n[bold cyan]TRACER Aggregation Scores[/bold cyan]")
        console.print(f"  Mean TRACER: {np.mean(all_tracer_scores):.4f}")
        console.print(f"  Std TRACER:  {np.std(all_tracer_scores):.4f}")
        console.print(f"  Min TRACER:  {np.min(all_tracer_scores):.4f}")
        console.print(f"  Max TRACER:  {np.max(all_tracer_scores):.4f}")
        console.print(f"  Total simulations: {len(all_tracer_scores)}")
        
        # Ground truth correlation
        if tracer_passed or tracer_failed:
            total_with_gt = len(tracer_passed) + len(tracer_failed)
            console.print(f"\n  Ground Truth: {len(tracer_passed)}/{total_with_gt} passed ({100*len(tracer_passed)/total_with_gt:.1f}%)")
            if tracer_passed:
                console.print(f"  Mean TRACER (passed): {np.mean(tracer_passed):.4f}")
            if tracer_failed:
                console.print(f"  Mean TRACER (failed): {np.mean(tracer_failed):.4f}")
            
            # Quick AUROC estimate (inline calculation for real-time summary)
            if SKLEARN_AVAILABLE and len(tracer_passed) > 0 and len(tracer_failed) > 0:
                try:
                    y_scores_inline = np.array(tracer_passed + tracer_failed)
                    y_true_inline = np.array([0]*len(tracer_passed) + [1]*len(tracer_failed))
                    auroc_inline = roc_auc_score(y_true_inline, y_scores_inline)
                    
                    auroc_color = "green" if auroc_inline > 0.7 else "yellow" if auroc_inline > 0.6 else "red"
                    console.print(f"\n  Quick AUROC: [{auroc_color}]{auroc_inline:.4f}[/{auroc_color}]")
                except Exception:
                    pass  # Silently skip if calculation fails
    
    # Per-simulation summary (first 3)
    console.print("\n[bold cyan]Per-Simulation Summary[/bold cyan] (showing first 3)\n")
    for i, sim in enumerate(results.simulations[:3]):
        agent_scores = []
        user_scores = []
        for msg in sim.messages:
            # Only process messages that have the uncertainty attribute
            if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                uncertainty_score = msg.uncertainty.get('normalized_entropy', 0.0)
                if msg.role == "assistant":
                    agent_scores.append(uncertainty_score)
                elif msg.role == "user":
                    user_scores.append(uncertainty_score)
        
        if agent_scores or user_scores:
            console.print(f"[bold]Simulation {i+1}[/bold] (Task: {sim.task_id})")
            console.print(f"  Turns: {len(agent_scores) + len(user_scores)}")
            if agent_scores:
                console.print(f"  Mean uncertainty (agent): {np.mean(agent_scores):.4f}")
            if user_scores:
                console.print(f"  Mean uncertainty (user):  {np.mean(user_scores):.4f}")
            console.print()
    
    if len(results.simulations) > 3:
        console.print(f"... and {len(results.simulations) - 3} more simulations\n")
    
    console.print("=" * 80, style="dim")
    console.print()


def print_summary(analysis: UncertaintyAnalysis, console: Console):
    """Print a summary of the analysis results."""
    console.print("\n" + "=" * 80, style="dim")
    console.print(
        "UNCERTAINTY ANALYSIS SUMMARY", style="bold cyan", justify="center"
    )
    console.print("=" * 80 + "\n", style="dim")

    # Metadata
    metadata = analysis.metadata
    console.print(f"[bold]Domain:[/bold] {metadata.get('domain', 'unknown')}")
    console.print(
        f"[bold]Agent Model:[/bold] {metadata.get('agent_llm', 'unknown')}"
    )
    console.print(f"[bold]User Model:[/bold] {metadata.get('user_llm', 'unknown')}")
    console.print(
        f"[bold]Total Simulations:[/bold] {metadata.get('total_simulations', 0)}\n"
    )

    # Overall statistics
    all_agent_uncertainties = []
    all_user_uncertainties = []

    for sim in analysis.results:
        for score in sim.uncertainty_scores:
            if score.actor == "agent":
                all_agent_uncertainties.append(score.normentropy_filtered_score)
            else:
                all_user_uncertainties.append(score.normentropy_filtered_score)

    if all_agent_uncertainties or all_user_uncertainties:
        console.print("[bold cyan]Overall Statistics[/bold cyan]")

        if all_agent_uncertainties:
            console.print("\n[bold]Agent Reasoning Uncertainty (U_i,agent):[/bold]")
            console.print(f"  Mean: {np.mean(all_agent_uncertainties):.4f}")
            console.print(f"  Std:  {np.std(all_agent_uncertainties):.4f}")
            console.print(f"  Min:  {np.min(all_agent_uncertainties):.4f}")
            console.print(f"  Max:  {np.max(all_agent_uncertainties):.4f}")

        if all_user_uncertainties:
            console.print("\n[bold]User Confusion (U_i,user):[/bold]")
            console.print(f"  Mean: {np.mean(all_user_uncertainties):.4f}")
            console.print(f"  Std:  {np.std(all_user_uncertainties):.4f}")
            console.print(f"  Min:  {np.min(all_user_uncertainties):.4f}")
            console.print(f"  Max:  {np.max(all_user_uncertainties):.4f}")
    
    # Semantic distance metrics
    all_da_scores = []
    all_do_scores = []
    all_do_agent = []
    all_do_user = []
    
    for sim in analysis.results:
        for score in sim.uncertainty_scores:
            if score.da_score is not None:
                all_da_scores.append(score.da_score)
            if score.do_score is not None:
                all_do_scores.append(score.do_score)
                if score.do_type == "agent_coherence":
                    all_do_agent.append(score.do_score)
                elif score.do_type == "user_coherence":
                    all_do_user.append(score.do_score)
    
    if all_da_scores or all_do_scores:
        console.print("\n[bold cyan]Situational Awareness Metrics[/bold cyan]")
        
        if all_da_scores:
            console.print("\n[bold]Local Repetition (Looping Penalty):[/bold]")
            console.print(f"  Mean: {np.mean(all_da_scores):.4f}")
            console.print(f"  Std:  {np.std(all_da_scores):.4f}")
            console.print(f"  Min:  {np.min(all_da_scores):.4f}")
            console.print(f"  Max:  {np.max(all_da_scores):.4f}")
            console.print(f"  Count: {len(all_da_scores)}")
            console.print(f"  [dim]High values indicate repetitive/looping behavior[/dim]")
        
        if all_do_scores:
            console.print("\n[bold]Inference Gap (Do):[/bold]")
            console.print(f"  Mean: {np.mean(all_do_scores):.4f}")
            console.print(f"  Std:  {np.std(all_do_scores):.4f}")
            console.print(f"  Min:  {np.min(all_do_scores):.4f}")
            console.print(f"  Max:  {np.max(all_do_scores):.4f}")
            console.print(f"  Count: {len(all_do_scores)}")
            
            if all_do_agent:
                console.print(f"\n  [dim]Agent Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_agent):.4f}")
                console.print(f"    Count: {len(all_do_agent)}")
            
            if all_do_user:
                console.print(f"\n  [dim]User Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_user):.4f}")
                console.print(f"    Count: {len(all_do_user)}")
    
    # TRACER Aggregation Scores
    all_tracer_scores = []
    tracer_passed = []
    tracer_failed = []
    
    for sim in analysis.results:
        if sim.tracer_metrics is not None:
            tracer_score = sim.tracer_metrics.get('tracer_score')
            if tracer_score is not None:
                all_tracer_scores.append(tracer_score)
                
                # Track by ground truth
                if sim.ground_truth_pass is not None:
                    if sim.ground_truth_pass:
                        tracer_passed.append(tracer_score)
                    else:
                        tracer_failed.append(tracer_score)
    
    if all_tracer_scores:
        console.print("\n[bold cyan]TRACER Aggregation Scores[/bold cyan]")
        console.print(f"  Mean TRACER: {np.mean(all_tracer_scores):.4f}")
        console.print(f"  Std TRACER:  {np.std(all_tracer_scores):.4f}")
        console.print(f"  Min TRACER:  {np.min(all_tracer_scores):.4f}")
        console.print(f"  Max TRACER:  {np.max(all_tracer_scores):.4f}")
        console.print(f"  Total simulations: {len(all_tracer_scores)}")
        
        # Ground truth correlation
        if tracer_passed or tracer_failed:
            total_with_gt = len(tracer_passed) + len(tracer_failed)
            console.print(f"\n  Ground Truth: {len(tracer_passed)}/{total_with_gt} passed ({100*len(tracer_passed)/total_with_gt:.1f}%)")
            if tracer_passed:
                console.print(f"  Mean TRACER (passed): {np.mean(tracer_passed):.4f}")
            if tracer_failed:
                console.print(f"  Mean TRACER (failed): {np.mean(tracer_failed):.4f}")
    
    # AUROC Evaluation (Failure Prediction)
    if analysis.auroc_metrics is not None:
        auroc = analysis.auroc_metrics
        console.print("\n[bold cyan]TRACER Evaluation (Failure Prediction)[/bold cyan]")
        console.print(f"  Hypothesis: High TRACER → High probability of failure")
        console.print(f"  Dataset: {auroc.num_samples} tasks ({auroc.num_failures} failures, {auroc.num_successes} successes)")
        console.print()
        
        # AUROC interpretation
        auroc_value = auroc.auroc
        if auroc_value > 0.9:
            auroc_color = "green"
            auroc_label = "Excellent"
        elif auroc_value > 0.8:
            auroc_color = "green"
            auroc_label = "Good"
        elif auroc_value > 0.7:
            auroc_color = "yellow"
            auroc_label = "Fair"
        elif auroc_value > 0.6:
            auroc_color = "yellow"
            auroc_label = "Poor"
        else:
            auroc_color = "red"
            auroc_label = "Very Poor"
        
        console.print(f"  AUROC: [{auroc_color}]{auroc_value:.4f}[/{auroc_color}] ({auroc_label})")
        console.print(f"  Optimal Threshold: {auroc.optimal_threshold:.4f}")
        console.print(f"  Accuracy: {auroc.accuracy:.4f}")
        console.print(f"  Precision: {auroc.precision:.4f}")
        console.print(f"  Recall: {auroc.recall:.4f}")
        console.print(f"  F1 Score: {auroc.f1_score:.4f}")
        
        # Interpretation
        console.print()
        if auroc_value > 0.7:
            console.print(f"  [green]✅ TRACER has {auroc_label.lower()} predictive power for task failure![/green]")
            console.print(f"  [green]   Tasks with TRACER > {auroc.optimal_threshold:.4f} are at high risk of failure.[/green]")
        elif auroc_value > 0.6:
            console.print(f"  [yellow]⚠️  TRACER shows fair predictive power (AUROC={auroc_value:.4f}).[/yellow]")
            console.print(f"  [yellow]   Consider combining with other features or tuning α,β,γ weights.[/yellow]")
        else:
            console.print(f"  [red]❌ TRACER shows poor predictive power (AUROC={auroc_value:.4f}).[/red]")
            if auroc.num_samples < 30:
                console.print(f"  [yellow]   Note: Small sample size ({auroc.num_samples}) limits statistical power.[/yellow]")
            else:
                console.print(f"  [yellow]   The hypothesis may not hold for this dataset/configuration.[/yellow]")
    
    # Baseline Comparison
    if analysis.baseline_aurocs is not None and len(analysis.baseline_aurocs) > 0:
        console.print("\n[bold cyan]Baseline Comparison[/bold cyan]")
        console.print("  Comparing TRACER against simpler baseline metrics:\n")
        
        # Create comparison table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white", width=35)
        table.add_column("AUROC", justify="right", width=10)
        table.add_column("Accuracy", justify="right", width=10)
        table.add_column("Precision", justify="right", width=10)
        table.add_column("Recall", justify="right", width=10)
        table.add_column("F1", justify="right", width=10)
        
        # Add TRACER as first row
        tracer_auroc_obj = analysis.auroc_metrics
        
        if tracer_auroc_obj is not None:
            table.add_row(
                "[bold]TRACER (Full Framework)[/bold]",
                f"[bold]{tracer_auroc_obj.auroc:.4f}[/bold]",
                f"[bold]{tracer_auroc_obj.accuracy:.4f}[/bold]",
                f"[bold]{tracer_auroc_obj.precision:.4f}[/bold]",
                f"[bold]{tracer_auroc_obj.recall:.4f}[/bold]",
                f"[bold]{tracer_auroc_obj.f1_score:.4f}[/bold]"
            )
            
            # Add baseline rows
            baseline_names = {
                'saup': 'SAUP (RMS Weighted Uncertainty)',
                'normalized_entropy': 'Normalized Entropy (U_i) Only',
                'self_assessed_confidence': 'Self-Assessed Confidence Only'
            }
            
            for baseline_key, baseline_metrics in analysis.baseline_aurocs.items():
                display_name = baseline_names.get(baseline_key, baseline_key)
                
                table.add_row(
                    display_name,
                    f"{baseline_metrics['auroc']:.4f}",
                    f"{baseline_metrics['accuracy']:.4f}",
                    f"{baseline_metrics['precision']:.4f}",
                    f"{baseline_metrics['recall']:.4f}",
                    f"{baseline_metrics['f1_score']:.4f}"
                )
            
            console.print(table)
            console.print()
            
            # Interpretation
            best_baseline_auroc = max(b['auroc'] for b in analysis.baseline_aurocs.values())
            best_baseline_key = max(
                analysis.baseline_aurocs.items(),
                key=lambda x: x[1]['auroc']
            )[0]
            best_baseline_name = baseline_names.get(best_baseline_key, best_baseline_key)
            
            improvement = tracer_auroc_obj.auroc - best_baseline_auroc
            
            if improvement > 0.05:
                console.print(f"  [green]✅ TRACER outperforms all baseline metrics[/green]")
                console.print(f"  [green]   Improvement over best baseline: +{improvement:.4f} AUROC points[/green]")
            elif improvement > 0:
                console.print(f"  [yellow]⚠️  TRACER shows modest improvement over best baseline (+{improvement:.4f})[/yellow]")
            else:
                console.print(f"  [red]❌ TRACER does not outperform best baseline[/red]")
                console.print(f"  [yellow]   Best baseline: {best_baseline_name} (AUROC={best_baseline_auroc:.4f})[/yellow]")
        else:
            console.print("  [yellow]⚠️  TRACER AUROC not available for comparison[/yellow]")
    
    # AUARC Evaluation (Selective Prediction)
    if analysis.auarc_metrics is not None:
        auarc = analysis.auarc_metrics
        console.print("\n[bold cyan]AUARC Evaluation (Selective Prediction)[/bold cyan]")
        console.print(f"  Hypothesis: Rejecting high-TRACER samples improves accuracy")
        console.print(f"  Dataset: {auarc.num_samples} tasks ({auarc.num_failures} failures, {auarc.num_successes} successes)")
        console.print()
        
        # AUARC interpretation
        auarc_value = auarc.auarc
        baseline_acc = auarc.baseline_accuracy
        max_acc = auarc.max_accuracy
        
        # Calculate potential improvement
        improvement = max_acc - baseline_acc
        
        if improvement > 0.1:
            auarc_color = "green"
            auarc_label = "Excellent selective prediction"
        elif improvement > 0.05:
            auarc_color = "green"
            auarc_label = "Good selective prediction"
        elif improvement > 0.02:
            auarc_color = "yellow"
            auarc_label = "Fair selective prediction"
        else:
            auarc_color = "red"
            auarc_label = "Poor selective prediction"
        
        console.print(f"  AUARC Score: [{auarc_color}]{auarc_value:.4f}[/{auarc_color}]")
        console.print(f"  Baseline Accuracy (no rejection): {baseline_acc:.4f}")
        console.print(f"  Max Accuracy (with rejection): [{auarc_color}]{max_acc:.4f}[/{auarc_color}]")
        console.print(f"  Potential Improvement: [{auarc_color}]+{improvement:.4f}[/{auarc_color}] ({auarc_label})")
        console.print()
        
        console.print(f"  Accuracy at 80% coverage: {auarc.accuracy_at_80_coverage:.4f}")
        console.print(f"  Accuracy at 50% coverage: {auarc.accuracy_at_50_coverage:.4f}")
        console.print(f"  Optimal coverage: {auarc.optimal_coverage:.1%} (accuracy: {auarc.optimal_accuracy:.4f})")
        
        # Interpretation
        console.print()
        if improvement > 0.05:
            console.print(f"  [green]✅ TRACER enables effective selective prediction![/green]")
            console.print(f"  [green]   By rejecting {(1-auarc.optimal_coverage)*100:.1f}% of high-uncertainty samples,[/green]")
            console.print(f"  [green]   accuracy improves from {baseline_acc:.1%} to {auarc.optimal_accuracy:.1%}.[/green]")
        elif improvement > 0.02:
            console.print(f"  [yellow]⚠️  TRACER shows modest selective prediction capability.[/yellow]")
            console.print(f"  [yellow]   Improvement: +{improvement:.1%} at {auarc.optimal_coverage:.1%} coverage.[/yellow]")
        else:
            console.print(f"  [red]❌ TRACER shows limited selective prediction capability.[/red]")
            console.print(f"  [yellow]   Rejecting uncertain samples yields minimal accuracy gains.[/yellow]")
    
    # Baseline Comparison for AUARC
    if analysis.baseline_auarcs is not None and len(analysis.baseline_auarcs) > 0:
        console.print("\n[bold cyan]AUARC Baseline Comparison[/bold cyan]")
        console.print("  Comparing TRACER against simpler baseline metrics for selective prediction:\n")
        
        # Create comparison table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white", width=35)
        table.add_column("AUARC", justify="right", width=10)
        table.add_column("Baseline Acc", justify="right", width=12)
        table.add_column("Max Acc", justify="right", width=10)
        table.add_column("Improvement", justify="right", width=12)
        table.add_column("Optimal Cov", justify="right", width=12)
        
        # Add TRACER as first row
        tracer_auarc_obj = analysis.auarc_metrics
        
        if tracer_auarc_obj is not None:
            tracer_improvement = tracer_auarc_obj.max_accuracy - tracer_auarc_obj.baseline_accuracy
            table.add_row(
                "[bold]TRACER (Full Framework)[/bold]",
                f"[bold]{tracer_auarc_obj.auarc:.4f}[/bold]",
                f"[bold]{tracer_auarc_obj.baseline_accuracy:.4f}[/bold]",
                f"[bold]{tracer_auarc_obj.max_accuracy:.4f}[/bold]",
                f"[bold]+{tracer_improvement:.4f}[/bold]",
                f"[bold]{tracer_auarc_obj.optimal_coverage:.1%}[/bold]"
            )
            
            # Add baseline rows
            baseline_names = {
                'saup': 'SAUP (RMS Weighted Uncertainty)',
                'normalized_entropy': 'Normalized Entropy (U_i) Only',
                'self_assessed_confidence': 'Self-Assessed Confidence Only'
            }
            
            for baseline_key, baseline_metrics in analysis.baseline_auarcs.items():
                display_name = baseline_names.get(baseline_key, baseline_key)
                baseline_improvement = baseline_metrics['max_accuracy'] - baseline_metrics['baseline_accuracy']
                
                table.add_row(
                    display_name,
                    f"{baseline_metrics['auarc']:.4f}",
                    f"{baseline_metrics['baseline_accuracy']:.4f}",
                    f"{baseline_metrics['max_accuracy']:.4f}",
                    f"+{baseline_improvement:.4f}",
                    f"{baseline_metrics['optimal_coverage']:.1%}"
                )
            
            console.print(table)
            console.print()
            
            # Interpretation
            best_baseline_improvement = max(
                b['max_accuracy'] - b['baseline_accuracy'] 
                for b in analysis.baseline_auarcs.values()
            )
            best_baseline_key = max(
                analysis.baseline_auarcs.items(),
                key=lambda x: x[1]['max_accuracy'] - x[1]['baseline_accuracy']
            )[0]
            best_baseline_name = baseline_names.get(best_baseline_key, best_baseline_key)
            
            improvement_diff = tracer_improvement - best_baseline_improvement
            
            if improvement_diff > 0.02:
                console.print(f"  [green]✅ TRACER outperforms all baseline metrics for selective prediction[/green]")
                console.print(f"  [green]   Additional improvement over best baseline: +{improvement_diff:.4f}[/green]")
            elif improvement_diff > 0:
                console.print(f"  [yellow]⚠️  TRACER shows modest improvement over best baseline (+{improvement_diff:.4f})[/yellow]")
            else:
                console.print(f"  [red]❌ TRACER does not outperform best baseline for selective prediction[/red]")
                console.print(f"  [yellow]   Best baseline: {best_baseline_name} (improvement: +{best_baseline_improvement:.4f})[/yellow]")
        else:
            console.print("  [yellow]⚠️  TRACER AUARC not available for comparison[/yellow]")

    # Early Warning Analysis
    if analysis.early_warning_metrics is not None and len(analysis.early_warning_metrics) > 0:
        console.print("\n[bold cyan]Early Warning Analysis (Threshold Crossing Detection)[/bold cyan]")
        console.print("  Analyzing when each metric first crosses its optimal failure threshold:\n")
        console.print("  [dim]Methodology: For each failed simulation, we find the FIRST turn where each metric[/dim]")
        console.print("  [dim]exceeds its optimal threshold (from AUROC analysis), then calculate at what[/dim]")
        console.print("  [dim]percentage of the trajectory this first crossing occurs.[/dim]")
        console.print("  [dim]Lower percentages indicate earlier warning signals.[/dim]")
        console.print("  [dim]Note: All metrics compared using their optimal discrimination thresholds.[/dim]")
        console.print("  [dim]      Self-Assessed Confidence converted to uncertainty via (1 - confidence).[/dim]\n")
        
        # Create comparison table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white", width=32)
        table.add_column("Mean", justify="right", width=8)
        table.add_column("Median", justify="right", width=8)
        table.add_column("<10%", justify="right", width=8)
        table.add_column("<20%", justify="right", width=8)
        table.add_column("<30%", justify="right", width=8)
        table.add_column("<40%", justify="right", width=8)
        table.add_column("<50%", justify="right", width=8)
        table.add_column("<60%", justify="right", width=8)
        table.add_column("<70%", justify="right", width=8)
        table.add_column("<80%", justify="right", width=8)
        table.add_column("<90%", justify="right", width=8)
        
        # Define order for metrics (TRACER first, then baselines)
        metric_order = ['tracer', 'saup', 'normalized_entropy', 'self_assessed_confidence']
        metric_display_names = {
            'tracer': 'TRACER',
            'saup': 'SAUP',
            'normalized_entropy': 'Normalized Entropy',
            'self_assessed_confidence': 'Self-Assessed Confidence'
        }
        
        # Collect data for comparison
        metric_data = []
        for metric_key in metric_order:
            if metric_key in analysis.early_warning_metrics:
                metric = analysis.early_warning_metrics[metric_key]
                metric_data.append({
                    'key': metric_key,
                    'name': metric_display_names.get(metric_key, metric_key),
                    'mean': metric.mean_peak_percentage,
                    'median': metric.median_peak_percentage,
                    'before_10': metric.percent_before_10,
                    'before_20': metric.percent_before_20,
                    'before_30': metric.percent_before_30,
                    'before_40': metric.percent_before_40,
                    'before_50': metric.percent_before_50,
                    'before_60': metric.percent_before_60,
                    'before_70': metric.percent_before_70,
                    'before_80': metric.percent_before_80,
                    'before_90': metric.percent_before_90
                })
        
        # Add rows to table
        for data in metric_data:
            # Highlight TRACER row
            name = f"[bold]{data['name']}[/bold]" if data['key'] == 'tracer' else data['name']
            
            # Color code mean first crossing position (lower is better - earlier warning)
            mean_val = data['mean']
            if mean_val < 30:
                mean_color = "green"
            elif mean_val < 50:
                mean_color = "yellow"
            else:
                mean_color = "red"
            
            # Color code percentages (higher is better - more tasks warned early)
            def color_percentage(pct: float) -> str:
                if pct > 60:
                    return "green"
                elif pct > 40:
                    return "yellow"
                else:
                    return "red"
            
            mean_str = f"[{mean_color}]{mean_val:.1f}%[/{mean_color}]" if data['key'] != 'tracer' else f"[bold {mean_color}]{mean_val:.1f}%[/bold {mean_color}]"
            median_str = f"{data['median']:.1f}%" if data['key'] != 'tracer' else f"[bold]{data['median']:.1f}%[/bold]"
            
            # Generate colored strings for all percentile columns
            percentile_strs = []
            for pct_key in ['before_10', 'before_20', 'before_30', 'before_40', 'before_50', 'before_60', 'before_70', 'before_80', 'before_90']:
                pct_val = data[pct_key]
                pct_color = color_percentage(pct_val)
                pct_str = f"[{pct_color}]{pct_val:.1f}%[/{pct_color}]" if data['key'] != 'tracer' else f"[bold {pct_color}]{pct_val:.1f}%[/bold {pct_color}]"
                percentile_strs.append(pct_str)
            
            table.add_row(
                name,
                mean_str,
                median_str,
                *percentile_strs  # Unpack all 9 percentile columns
            )
        
        console.print(table)
        console.print()
        
        # Interpretation
        if 'tracer' in analysis.early_warning_metrics:
            tracer_metric = analysis.early_warning_metrics['tracer']
            
            # Compare TRACER against baselines
            baseline_means = [
                data['mean'] for data in metric_data if data['key'] != 'tracer'
            ]
            
            if baseline_means:
                best_baseline_mean = min(baseline_means)
                best_baseline_key = next(
                    data['key'] for data in metric_data 
                    if data['key'] != 'tracer' and data['mean'] == best_baseline_mean
                )
                best_baseline_name = metric_display_names.get(best_baseline_key, best_baseline_key)
                
                improvement = best_baseline_mean - tracer_metric.mean_peak_percentage
                
                console.print(f"  [bold]Num. Failed Tasks Analyzed:[/bold] {tracer_metric.num_failed_tasks}")
                console.print()
                
                if improvement > 10:
                    console.print(f"  [green]✅ TRACER signals failure significantly earlier than baselines![/green]")
                    console.print(f"  [green]   TRACER crosses threshold at {tracer_metric.mean_peak_percentage:.1f}% vs. best baseline at {best_baseline_mean:.1f}%[/green]")
                    console.print(f"  [green]   Improvement: {improvement:.1f} percentage points earlier[/green]")
                elif improvement > 5:
                    console.print(f"  [green]✅ TRACER signals failure earlier than baselines[/green]")
                    console.print(f"  [green]   TRACER crosses threshold at {tracer_metric.mean_peak_percentage:.1f}% vs. best baseline at {best_baseline_mean:.1f}%[/green]")
                    console.print(f"  [green]   Improvement: {improvement:.1f} percentage points earlier[/green]")
                elif improvement > 0:
                    console.print(f"  [yellow]⚠️  TRACER shows modest improvement in early warning (+{improvement:.1f} pp)[/yellow]")
                    console.print(f"  [yellow]   TRACER crosses threshold at {tracer_metric.mean_peak_percentage:.1f}% vs. {best_baseline_name} at {best_baseline_mean:.1f}%[/yellow]")
                else:
                    console.print(f"  [red]❌ TRACER does not signal earlier than best baseline[/red]")
                    console.print(f"  [yellow]   Best baseline: {best_baseline_name} (crosses at {best_baseline_mean:.1f}%)[/yellow]")
                    console.print(f"  [yellow]   TRACER crosses threshold at {tracer_metric.mean_peak_percentage:.1f}%[/yellow]")
                
                # Additional insights
                console.print()
                if tracer_metric.percent_before_20 > 50:
                    console.print(f"  [green]🎯 TRACER provides early warning (before 20% of trajectory) in {tracer_metric.percent_before_20:.1f}% of failures[/green]")
                elif tracer_metric.percent_before_40 > 50:
                    console.print(f"  [yellow]🎯 TRACER provides moderate early warning (before 40% of trajectory) in {tracer_metric.percent_before_40:.1f}% of failures[/yellow]")
                else:
                    console.print(f"  [yellow]⚠️  TRACER warning signals tend to appear later in trajectories[/yellow]")

    # Ablation Studies
    if analysis.ablation_studies is not None and len(analysis.ablation_studies) > 0:
        console.print("\n[bold cyan]Ablation Studies[/bold cyan]")
        console.print("  Analyzing impact of isolating assistant vs. user turns:\n")
        
        for ablation_key, ablation_data in analysis.ablation_studies.items():
            ablation_auroc = ablation_data.get('auroc_metrics')
            ablation_baselines = ablation_data.get('baseline_aurocs')
            description = ablation_data.get('description', ablation_key)
            
            console.print(f"\n[bold white]{description}[/bold white]")
            
            if ablation_auroc or ablation_baselines:
                # Create ablation table
                ablation_table = Table(show_header=True, header_style="bold magenta")
                ablation_table.add_column("Metric", style="white", width=35)
                ablation_table.add_column("AUROC", justify="right", width=10)
                ablation_table.add_column("Accuracy", justify="right", width=10)
                ablation_table.add_column("Precision", justify="right", width=10)
                ablation_table.add_column("Recall", justify="right", width=10)
                ablation_table.add_column("F1", justify="right", width=10)
                
                # Add TRACER row if available
                if ablation_auroc is not None:
                    ablation_table.add_row(
                        "[bold]TRACER[/bold]",
                        f"[bold]{ablation_auroc.auroc:.4f}[/bold]",
                        f"[bold]{ablation_auroc.accuracy:.4f}[/bold]",
                        f"[bold]{ablation_auroc.precision:.4f}[/bold]",
                        f"[bold]{ablation_auroc.recall:.4f}[/bold]",
                        f"[bold]{ablation_auroc.f1_score:.4f}[/bold]"
                    )
                
                # Add baseline rows
                if ablation_baselines is not None:
                    baseline_names = {
                        'saup': 'SAUP',
                        'normalized_entropy': 'Normalized Entropy',
                        'self_assessed_confidence': 'Self-Assessed Confidence'
                    }
                    
                    for baseline_key, baseline_metrics in ablation_baselines.items():
                        display_name = baseline_names.get(baseline_key, baseline_key)
                        
                        ablation_table.add_row(
                            display_name,
                            f"{baseline_metrics['auroc']:.4f}",
                            f"{baseline_metrics['accuracy']:.4f}",
                            f"{baseline_metrics['precision']:.4f}",
                            f"{baseline_metrics['recall']:.4f}",
                            f"{baseline_metrics['f1_score']:.4f}"
                        )
                
                console.print(ablation_table)
                console.print()
                
                # Compare with full analysis
                if analysis.auroc_metrics is not None and ablation_auroc is not None:
                    full_auroc = analysis.auroc_metrics.auroc
                    ablation_auroc_val = ablation_auroc.auroc
                    diff = ablation_auroc_val - full_auroc
                    
                    if abs(diff) < 0.02:
                        console.print(f"  → Similar performance to full analysis (Δ={diff:+.4f})")
                    elif diff > 0:
                        console.print(f"  [green]→ Better than full analysis (Δ={diff:+.4f})[/green]")
                    else:
                        console.print(f"  [yellow]→ Worse than full analysis (Δ={diff:+.4f})[/yellow]")
            else:
                console.print("  [yellow]⚠️  No metrics available for this ablation[/yellow]")

    # Per-simulation summary
    console.print("\n[bold cyan]Per-Simulation Summary[/bold cyan]\n")
    for i, sim in enumerate(analysis.results[:5]):  # Show first 5
        summary = sim.summary
        console.print(f"[bold]Simulation {i+1}[/bold] (Task: {sim.task_id})")
        console.print(f"  Turns: {sim.turn_count}")
        console.print(
            f"  Mean uncertainty (overall): {summary.get('mean_uncertainty_overall', 0):.4f}"
        )
        console.print(
            f"  Mean uncertainty (agent):   {summary.get('mean_uncertainty_agent', 0):.4f}"
        )
        console.print(
            f"  Mean uncertainty (user):    {summary.get('mean_uncertainty_user', 0):.4f}"
        )
        
        # Display TRACER score if available
        if sim.tracer_metrics:
            console.print(f"  TRACER Score: {sim.tracer_metrics['tracer_score']:.4f}")
            console.print(f"  Ground Truth: {'✅ Pass' if sim.ground_truth_pass else '❌ Fail' if sim.ground_truth_pass is not None else 'N/A'}\n")
        else:
            console.print()

    if len(analysis.results) > 5:
        console.print(f"... and {len(analysis.results) - 5} more simulations\n")

    console.print("=" * 80, style="dim")


def print_detailed_trajectory(
    analysis: UncertaintyAnalysis, sim_index: int, console: Console
):
    """Print detailed turn-by-turn uncertainty for a specific simulation."""
    if sim_index >= len(analysis.results):
        console.print(
            f"[yellow]⚠️  Simulation index {sim_index} not found.[/yellow]"
        )
        return

    sim = analysis.results[sim_index]

    console.print("\n" + "=" * 80, style="dim")
    console.print(
        f"DETAILED TRAJECTORY: Simulation {sim_index + 1}",
        style="bold cyan",
        justify="center",
    )
    console.print("=" * 80, style="dim")
    console.print(f"\n[bold]Simulation ID:[/bold] {sim.simulation_id}")
    console.print(f"[bold]Task ID:[/bold] {sim.task_id}")
    console.print(f"[bold]Trial:[/bold] {sim.trial}")
    console.print(f"[bold]Total Turns:[/bold] {sim.turn_count}")
    
    # Display TRACER score if available
    if sim.tracer_metrics:
        console.print(f"[bold]TRACER Score:[/bold] {sim.tracer_metrics['tracer_score']:.4f}")
        console.print(f"[bold]Ground Truth:[/bold] {'✅ Pass' if sim.ground_truth_pass else '❌ Fail' if sim.ground_truth_pass is not None else 'N/A'}")
    console.print()

    # Calculate penalties for display
    penalties = []
    if sim.tracer_metrics:
        config = TRACERConfig()
        for score in sim.uncertainty_scores:
            from tau2.metrics.uncertainty import calculate_situational_weight
            penalty = calculate_situational_weight(
                score.da_score,
                score.do_score if score.do_type == "agent_coherence" else None,
                score.do_score if score.do_type == "user_coherence" else None,
                config
            )
            penalties.append(penalty)

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Turn", style="dim", width=5)
    table.add_column("Actor", width=8)
    table.add_column("U_i", justify="right", width=8)
    table.add_column("Repeat", justify="right", width=8)
    table.add_column("Do", justify="right", width=8)
    table.add_column("Penalty", justify="right", width=8)
    table.add_column("Do Type", width=10)
    table.add_column("Content", width=35)

    for idx, score in enumerate(sim.uncertainty_scores):
        # Color code by uncertainty level
        if score.normentropy_filtered_score < 0.1:
            ui_color = "green"
        elif score.normentropy_filtered_score < 0.3:
            ui_color = "yellow"
        else:
            ui_color = "red"
        
        # Color code Repetition (da_score now stores repetition)
        da_str = "—"
        if score.da_score is not None:
            # High repetition is bad (red), low is good (green)
            if score.da_score < 0.3:
                da_color = "green"
            elif score.da_score < 0.7:
                da_color = "yellow"
            else:
                da_color = "red"
            da_str = f"[{da_color}]{score.da_score:.3f}[/{da_color}]"
        
        # Color code Do (inference gap)
        do_str = "—"
        if score.do_score is not None:
            if score.do_score < 0.3:
                do_color = "green"
            elif score.do_score < 0.6:
                do_color = "yellow"
            else:
                do_color = "red"
            do_str = f"[{do_color}]{score.do_score:.3f}[/{do_color}]"
        
        # Display penalty if available
        penalty_str = "—"
        if penalties and idx < len(penalties):
            penalty = penalties[idx]
            if penalty < 0.3:
                penalty_color = "green"
            elif penalty < 0.6:
                penalty_color = "yellow"
            else:
                penalty_color = "red"
            penalty_str = f"[{penalty_color}]{penalty:.3f}[/{penalty_color}]"
        
        do_type_str = score.do_type[:10] if score.do_type else "—"

        table.add_row(
            str(score.turn),
            score.actor,
            f"[{ui_color}]{score.normentropy_filtered_score:.3f}[/{ui_color}]",
            da_str,
            do_str,
            penalty_str,
            do_type_str,
            score.content_preview[:35] + "...",
        )

    console.print(table)
    console.print("\n" + "=" * 80, style="dim")


def main():
    """Main entry point for the uncertainty analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze Tau-2 trajectories and calculate uncertainty metrics"
    )
    parser.add_argument(
        "simulation_file", type=str, help="Path to the simulation results JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (optional, defaults to data/uncertainty/<same_filename>)",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include detailed statistics in output",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Print detailed turn-by-turn trajectory",
    )
    parser.add_argument(
        "--sim-index",
        type=int,
        default=0,
        help="Simulation index to show in detailed view (default: 0)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )
    parser.add_argument(
        "--tracer-config",
        type=json.loads,
        default='{"alpha": 4.0, "beta": 4.0, "gamma": 5.0, "top_k_percentile": 0.26, "ensemble_weight_max": 0.2}',
        help='TRACER weight configuration (JSON format, e.g., \'{"alpha": 4.0, "beta": 4.0, "gamma": 5.0, "top_k_percentile": 0.26, "ensemble_weight_max": 0.2}\')',
    )
    parser.add_argument(
        "--no-auroc",
        action="store_true",
        help="Skip AUROC calculation (useful if ground truth not available)",
    )

    args = parser.parse_args()

    console = Console()

    # Check if file exists
    sim_path = Path(args.simulation_file)
    if not sim_path.exists():
        console.print(
            f"[red]❌ Error: File not found: {args.simulation_file}[/red]"
        )
        sys.exit(1)

    try:
        # Load simulation data
        console.print(
            f"[cyan]📂 Loading simulation results from: {args.simulation_file}[/cyan]"
        )
        results = Results.load(sim_path)

        # Parse TRACER configuration from CLI arguments
        tracer_config = TRACERConfig(**args.tracer_config)
        console.print(f"[cyan]⚙️  TRACER Config: α={tracer_config.alpha}, β={tracer_config.beta}, γ={tracer_config.gamma}[/cyan]")

        # Analyze
        console.print("[cyan]🔄 Analyzing trajectories and calculating uncertainty scores...[/cyan]")
        analysis = analyze_results(
            results,
            config=tracer_config,
            verbose=args.verbose,
            calculate_auroc=not args.no_auroc
        )

        # Print summary
        print_summary(analysis, console)

        # Print detailed trajectory if requested
        if args.detailed:
            print_detailed_trajectory(analysis, args.sim_index, console)

        # Determine output path
        if not args.no_save:
            if args.output:
                # User specified custom output path
                output_path = Path(args.output)
            else:
                # Default: save to data/uncertainty with same filename
                sim_filename = sim_path.name
                # Get the project root (assuming script is in src/tau2/scripts/)
                project_root = Path(__file__).parent.parent.parent.parent
                output_dir = project_root / "data" / "uncertainty"
                output_path = output_dir / sim_filename
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the analysis
            with open(output_path, "w") as f:
                json.dump(analysis.model_dump(), f, indent=2)

            console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")

        console.print("\n[green]✅ Analysis complete![/green]")

    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Error: Invalid JSON file: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error processing file: {e}[/red]")
        logger.exception("Error during uncertainty analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()

