"""
TRACER Parameter Optimization Script (MAX Variant)

Systematically searches for optimal TRACER parameters (alpha, beta, gamma, 
top_k_percentile, ensemble_weight_max) that maximize AUROC for failure prediction.

Uses MAX variant formula: risk_i = max(U_i, α·Da_i, β·Do_agent_i, γ·Do_user_i)

This script implements a multi-stage grid search strategy:
1. Coarse Grid Search: Explores wide parameter ranges to find promising regions
2. Fine-Grained Search: Refines parameters around the best coarse solution
3. Ensemble Optimization: Tunes the ensemble weight for combining top-k and max signals

The script accepts either:
- A single JSON file containing multiple simulations
- A directory containing multiple JSON files (will merge all simulations)

By default, results are automatically saved to data/optimization/ with the same 
filename as the input file for easy cross-referencing.

Usage:
    # AUTO MODE (Recommended): Run all 3 stages sequentially
    # Results auto-save to data/optimization/my_simulation_results.json
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json
    
    # Or using a directory of simulation files
    python -m tau2.scripts.optimize_tracer_parameters data/simulations/
    
    # Custom output location
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --output results/custom_optimization.json
    
    # Don't save, just display results
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --no-save
    
    # Stage 1: Coarse grid search only
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode coarse
    
    # Stage 2: Fine-grained search (around best coarse params)
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode fine \\
        --alpha-center 7.0 --alpha-range 3.0 --alpha-step 0.5 \\
        --topk-center 0.25 --topk-range 0.1 --topk-step 0.01
    
    # Stage 3: Ensemble optimization
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode ensemble \\
        --alpha 4.0 --beta 4.0 --gamma 5.0 --top-k 0.26 \\
        --ensemble-weights 0.05 0.1 0.15 0.2 0.25 0.3
    
    # Custom grid search
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode custom \\
        --alpha-values 3.0 4.0 5.0 \\
        --beta-values 3.0 4.0 5.0 \\
        --gamma-values 4.0 5.0 6.0 \\
        --topk-values 0.2 0.25 0.3 \\
        --ensemble-values 0.15 0.2 0.25
    
    # Random search (explore parameter space randomly - good for global optima)
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode random \\
        --n-random 1000
    
    # Wide search (broad grid + random exploration - most thorough)
    python -m tau2.scripts.optimize_tracer_parameters \\
        data/simulations/my_simulation_results.json \\
        --mode wide \\
        --n-random 1000

Example Output (Auto Mode):
    Stage 1 (Coarse):   AUROC = 0.6856 (1125 configs)
    Stage 2 (Fine):     AUROC = 0.6890 (315 configs, +0.0034)
    Stage 3 (Ensemble): AUROC = 0.7007 (9 configs, +0.0117)
    
    Total Improvement: +0.0151 (2.2%)
    
    Best Configuration:
      Alpha (α): 4.0
      Beta (β): 4.0
      Gamma (γ): 5.0
      Top-K Percentile: 0.26
      Ensemble Weight (Max): 0.20
      
      AUROC: 0.7007
    
    ✓ Results saved to data/optimization/my_simulation_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
import itertools
from datetime import datetime

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from tau2.data_model.simulation import Results
from tau2.metrics.uncertainty import TRACERConfig, calculate_normalized_entropy
from tau2.scripts.analyze_uncertainty import analyze_results, calculate_auroc_metrics

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. AUROC evaluation will be disabled.")

import random


def load_simulations_from_path(sim_path: Path) -> list[Results]:
    """
    Load simulation JSON file(s) from a path (file or directory).
    
    Args:
        sim_path: Path to a single JSON file or directory containing JSON files
        
    Returns:
        List of Results objects
    """
    sim_files = []
    
    # Check if path is a file or directory
    if sim_path.is_file():
        if sim_path.suffix == '.json':
            sim_files = [sim_path]
        else:
            logger.error(f"File must be a JSON file: {sim_path}")
            sys.exit(1)
    elif sim_path.is_dir():
        sim_files = list(sim_path.glob("*.json"))
        if not sim_files:
            logger.error(f"No JSON files found in directory: {sim_path}")
            sys.exit(1)
        logger.info(f"Found {len(sim_files)} JSON files in {sim_path}")
    else:
        logger.error(f"Path does not exist: {sim_path}")
        sys.exit(1)
    
    all_results = []
    for sim_file in sim_files:
        try:
            with open(sim_file, 'r') as f:
                data = json.load(f)
            results = Results.model_validate(data)
            all_results.append(results)
            logger.info(f"Loaded {sim_file.name}: {len(results.simulations)} simulations")
        except Exception as e:
            logger.warning(f"Failed to load {sim_file.name}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid simulation files could be loaded")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(all_results)} file(s)")
    return all_results


def merge_results(results_list: list[Results]) -> Results:
    """
    Merge multiple Results objects into a single Results object.
    
    Args:
        results_list: List of Results objects to merge
        
    Returns:
        Single merged Results object
    """
    if not results_list:
        raise ValueError("Cannot merge empty results list")
    
    # Use first result as base
    merged = results_list[0].model_copy(deep=True)
    
    # Merge all simulations
    for results in results_list[1:]:
        merged.simulations.extend(results.simulations)
    
    # Update metadata
    merged.timestamp = datetime.now().isoformat()
    
    return merged


def precompute_step_metrics(merged_results: Results, console: Console) -> list[dict]:
    """
    Phase 1: Pre-compute all step-level metrics (U_i, Da, Do) that are independent 
    of alpha/beta/gamma/top_k/ensemble parameters.
    
    This function extracts and caches the expensive computations (especially embeddings)
    that are properties of the simulation data itself and don't change across different
    parameter configurations.
    
    Args:
        merged_results: Merged simulation results
        console: Rich console for progress display
        
    Returns:
        List of pre-computed simulation data, where each item contains:
        {
            'steps': List of step dicts with {'ui', 'da', 'do_agent', 'do_user'},
            'ground_truth_pass': bool (True if task succeeded)
        }
    """
    from tau2.metrics.uncertainty import (
        calculate_hybrid_repetition_score,
        calculate_tool_repetition
    )
    
    console.print("[bold cyan]Phase 1: Pre-computing step-level metrics (U_i, Da, Do)...[/bold cyan]")
    console.print(f"Processing {len(merged_results.simulations)} simulations...")
    
    precomputed_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Pre-computing metrics...", total=len(merged_results.simulations))
        
        for simulation in merged_results.simulations:
            sim_dict = simulation.model_dump()
            
            # Extract step-level metrics
            step_data = []
            
            # Track agent history for repetition detection
            agent_text_history = []
            agent_tool_history = []
            
            for message in sim_dict.get("messages", []):
                role = message.get("role")
                
                # Only process agent and user messages
                if role not in ["assistant", "user"]:
                    continue
                
                # Calculate U_i (normalized entropy) - INDEPENDENT of alpha/beta/gamma
                logprobs = message.get("logprobs")
                ui_score = calculate_normalized_entropy(logprobs)
                
                # Calculate Da (repetition) - INDEPENDENT of alpha/beta/gamma
                da_score = None
                if role == "assistant":
                    content = message.get("content", "")
                    if content is None:
                        content = ""
                    
                    # Text-based hybrid repetition
                    text_repetition = 0.0
                    if content:
                        text_repetition = calculate_hybrid_repetition_score(
                            content, 
                            agent_text_history
                        )
                        agent_text_history.append(content)
                    
                    # Tool-based repetition
                    tool_repetition = 0.0
                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        tool_repetition = calculate_tool_repetition(
                            tool_calls,
                            agent_tool_history
                        )
                        agent_tool_history.append(tool_calls)
                    
                    # Aggregate: max of text and tool repetition
                    da_score = max(text_repetition, tool_repetition)
                
                # Extract Do (inference gap) - INDEPENDENT of alpha/beta/gamma
                do_score = message.get("do_score")
                do_type = message.get("do_type")
                
                # Store pre-computed step metrics
                step_data.append({
                    'ui': ui_score,
                    'da': da_score,
                    'do_agent': do_score if do_type == "agent_coherence" else None,
                    'do_user': do_score if do_type == "user_coherence" else None
                })
            
            # Extract ground truth
            ground_truth = sim_dict.get("reward_info", {}).get("reward", None) if sim_dict.get("reward_info") else None
            ground_truth_pass = ground_truth == 1.0 if ground_truth is not None else None
            
            precomputed_data.append({
                'steps': step_data,
                'ground_truth_pass': ground_truth_pass
            })
            
            progress.update(task, advance=1)
    
    console.print(f"[green]✓ Pre-computation complete![/green]\n")
    return precomputed_data


def evaluate_config_fast(
    config: TRACERConfig,
    precomputed_data: list[dict],
    verbose: bool = False
) -> Optional[float]:
    """
    Phase 2: Fast evaluation using pre-computed step metrics.
    
    This function only performs the lightweight arithmetic operations:
    - step_risk = max(U_i, alpha*Da, beta*Do_agent, gamma*Do_user)  [MAX variant]
    - top-k filtering
    - ensemble aggregation
    - AUROC calculation
    
    No expensive operations (embeddings, entropy calculations) are performed here.
    
    Args:
        config: TRACER configuration to evaluate
        precomputed_data: Pre-computed step metrics from Phase 1
        verbose: If True, print detailed logs
        
    Returns:
        AUROC score, or None if evaluation failed
    """
    if not SKLEARN_AVAILABLE:
        if verbose:
            logger.warning("scikit-learn not available")
        return None
    
    try:
        # Calculate TRACER scores for each simulation
        y_scores = []
        y_true = []
        
        for sim_data in precomputed_data:
            steps = sim_data['steps']
            ground_truth_pass = sim_data['ground_truth_pass']
            
            # Skip simulations without ground truth
            if ground_truth_pass is None or not steps:
                continue
            
            # Calculate step risks using MAX variant formula
            step_risks = []
            for step in steps:
                ui = step['ui']
                da = step['da'] if step['da'] is not None else 0.0
                do_agent = step['do_agent'] if step['do_agent'] is not None else 0.0
                do_user = step['do_user'] if step['do_user'] is not None else 0.0
                
                # MAX formula: risk = max(U_i, alpha*Da, beta*Do_agent, gamma*Do_user)
                step_risk = max(
                    ui,
                    config.alpha * da,
                    config.beta * do_agent,
                    config.gamma * do_user
                )
                step_risks.append(step_risk)
            
            if not step_risks:
                continue
            
            # Top-k aggregation
            if config.top_k_percentile >= 1.0:
                top_k_risks = step_risks
            else:
                sorted_risks = sorted(step_risks, reverse=True)
                top_k_count = max(1, int(config.top_k_percentile * len(step_risks)))
                top_k_risks = sorted_risks[:top_k_count]
            
            # Calculate mean of top-k
            mean_top_k = float(np.mean(top_k_risks))
            
            # Ensemble: combine top-k mean with max risk
            if config.ensemble_weight_max > 0.0:
                max_risk = float(np.max(step_risks))
                tracer_score = (1 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk
            else:
                tracer_score = mean_top_k
            
            # Store for AUROC calculation
            y_scores.append(tracer_score)
            # Label encoding: Failure=1, Success=0
            y_true.append(0 if ground_truth_pass else 1)
        
        # Calculate AUROC
        if len(y_scores) < 2:
            if verbose:
                logger.warning(f"Not enough samples for AUROC ({len(y_scores)} < 2)")
            return None
        
        y_scores_arr = np.array(y_scores)
        y_true_arr = np.array(y_true)
        
        unique_labels = np.unique(y_true_arr)
        if len(unique_labels) < 2:
            if verbose:
                logger.warning(f"Only one class present: {unique_labels}")
            return None
        
        auroc = roc_auc_score(y_true_arr, y_scores_arr)
        return float(auroc)
    
    except Exception as e:
        if verbose:
            logger.error(f"Failed to evaluate config: {e}")
        return None


def evaluate_config(
    config: TRACERConfig,
    merged_results: Results,
    verbose: bool = False
) -> Optional[float]:
    """
    Legacy evaluation function (slower, but complete).
    
    This function is kept for backward compatibility and non-optimization use cases.
    For optimization, use precompute_step_metrics() + evaluate_config_fast() instead.
    
    Args:
        config: TRACER configuration to evaluate
        merged_results: Merged simulation results
        verbose: If True, print detailed logs
        
    Returns:
        AUROC score, or None if evaluation failed
    """
    try:
        analysis = analyze_results(
            merged_results,
            config=config,
            verbose=False,
            calculate_auroc=True
        )
        
        if analysis.auroc_metrics is None:
            if verbose:
                logger.warning("AUROC calculation failed for config")
            return None
        
        return analysis.auroc_metrics.auroc
    
    except Exception as e:
        if verbose:
            logger.error(f"Failed to evaluate config: {e}")
        return None


def coarse_grid_search(
    merged_results: Results,
    console: Console,
    precomputed_data: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Stage 1: Coarse grid search over wide parameter ranges.
    
    Search space (updated to include 0.0 for all parameters):
        - alpha: [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        - beta: [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        - gamma: [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        - top_k_percentile: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        - ensemble_weight_max: [0.0, 0.1, 0.2, 0.3]
    
    Note: Including 0.0 is critical as diagnostic analysis showed that
    α=0.0 and γ=0.0 are often optimal (disabling Da and Do_user penalties).
    
    Args:
        merged_results: Merged simulation results (used only if precomputed_data is None)
        console: Rich console for output
        precomputed_data: Pre-computed step metrics (if None, will use legacy evaluation)
    
    Returns:
        Dictionary with best configuration and results
    """
    alpha_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    beta_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    gamma_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    topk_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    ensemble_values = [0.0, 0.1, 0.2, 0.3]
    
    configs = list(itertools.product(alpha_values, beta_values, gamma_values, topk_values, ensemble_values))
    total_configs = len(configs)
    
    console.print(f"\n[bold cyan]Stage 1: Coarse Grid Search[/bold cyan]")
    console.print(f"Testing {total_configs} configurations...")
    console.print(f"  Alpha: {len(alpha_values)} values")
    console.print(f"  Beta: {len(beta_values)} values")
    console.print(f"  Gamma: {len(gamma_values)} values")
    console.print(f"  Top-K: {len(topk_values)} values")
    console.print(f"  Ensemble: {len(ensemble_values)} values\n")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing...", total=total_configs)
        
        for alpha, beta, gamma, topk, ensemble in configs:
            config = TRACERConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k_percentile=topk,
                ensemble_weight_max=ensemble
            )
            
            # Use fast evaluation if precomputed data available, otherwise fall back to legacy
            if precomputed_data is not None:
                auroc = evaluate_config_fast(config, precomputed_data)
            else:
                auroc = evaluate_config(config, merged_results)
            
            if auroc is not None:
                results.append({
                    'config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'top_k_percentile': topk,
                        'ensemble_weight_max': ensemble
                    },
                    'auroc': auroc
                })
            
            progress.update(task, advance=1)
    
    # Sort by AUROC
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    return {
        'mode': 'coarse',
        'total_configurations': total_configs,
        'successful_evaluations': len(results),
        'best_config': results[0]['config'],
        'best_auroc': results[0]['auroc'],
        'top_10_configs': results[:10]
    }


def random_search(
    merged_results: Results,
    console: Console,
    n_iterations: int = 500,
    alpha_range: tuple[float, float] = (0.0, 20.0),
    beta_range: tuple[float, float] = (0.0, 20.0),
    gamma_range: tuple[float, float] = (0.0, 20.0),
    topk_range: tuple[float, float] = (0.05, 1.0),
    ensemble_range: tuple[float, float] = (0.0, 0.5),
    precomputed_data: Optional[list[dict]] = None,
    seed: Optional[int] = None
) -> dict[str, Any]:
    """
    Random search over continuous parameter space.
    
    Often more effective than grid search for finding global optima because:
    - Explores diverse regions of parameter space
    - Not biased by grid structure
    - Can test extreme parameter combinations
    
    Args:
        merged_results: Merged simulation results
        console: Rich console for output
        n_iterations: Number of random configurations to test
        alpha_range: (min, max) for alpha parameter
        beta_range: (min, max) for beta parameter
        gamma_range: (min, max) for gamma parameter  
        topk_range: (min, max) for top_k_percentile
        ensemble_range: (min, max) for ensemble_weight_max
        precomputed_data: Pre-computed step metrics
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with best configuration and results
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    console.print(f"\n[bold cyan]Random Search[/bold cyan]")
    console.print(f"Testing {n_iterations} random configurations...")
    console.print(f"  Alpha: [{alpha_range[0]}, {alpha_range[1]}]")
    console.print(f"  Beta: [{beta_range[0]}, {beta_range[1]}]")
    console.print(f"  Gamma: [{gamma_range[0]}, {gamma_range[1]}]")
    console.print(f"  Top-K: [{topk_range[0]}, {topk_range[1]}]")
    console.print(f"  Ensemble: [{ensemble_range[0]}, {ensemble_range[1]}]\n")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Exploring...", total=n_iterations)
        
        for _ in range(n_iterations):
            # Sample random parameters from uniform distributions
            alpha = random.uniform(*alpha_range)
            beta = random.uniform(*beta_range)
            gamma = random.uniform(*gamma_range)
            topk = random.uniform(*topk_range)
            ensemble = random.uniform(*ensemble_range)
            
            config = TRACERConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k_percentile=topk,
                ensemble_weight_max=ensemble
            )
            
            # Evaluate
            if precomputed_data is not None:
                auroc = evaluate_config_fast(config, precomputed_data)
            else:
                auroc = evaluate_config(config, merged_results)
            
            if auroc is not None:
                results.append({
                    'config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'top_k_percentile': topk,
                        'ensemble_weight_max': ensemble
                    },
                    'auroc': auroc
                })
            
            progress.update(task, advance=1)
    
    # Sort by AUROC
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    return {
        'mode': 'random',
        'total_configurations': n_iterations,
        'successful_evaluations': len(results),
        'best_config': results[0]['config'] if results else None,
        'best_auroc': results[0]['auroc'] if results else None,
        'top_10_configs': results[:10]
    }


def fine_grained_search(
    merged_results: Results,
    console: Console,
    alpha_center: float = 7.0,
    alpha_range: float = 3.0,
    alpha_step: float = 0.5,
    topk_center: float = 0.25,
    topk_range: float = 0.1,
    topk_step: float = 0.01,
    beta: float = 4.0,
    gamma: float = 5.0,
    ensemble: float = 0.15,
    precomputed_data: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Stage 2: Fine-grained search around best coarse parameters.
    
    Args:
        merged_results: Merged simulation results (used only if precomputed_data is None)
        console: Rich console for output
        alpha_center: Center value for alpha search
        alpha_range: Range around center (±range)
        alpha_step: Step size for alpha
        topk_center: Center value for top_k search
        topk_range: Range around center (±range)
        topk_step: Step size for top_k
        beta: Fixed beta value
        gamma: Fixed gamma value
        ensemble: Fixed ensemble weight
        precomputed_data: Pre-computed step metrics (if None, will use legacy evaluation)
        
    Returns:
        Dictionary with best configuration and results
    """
    # Generate alpha values (ensure non-negative)
    alpha_min = max(0.0, alpha_center - alpha_range)
    alpha_max = alpha_center + alpha_range
    alpha_values = np.arange(alpha_min, alpha_max + alpha_step, alpha_step).tolist()
    
    # Generate top_k values (ensure valid range [0.05, 1.0])
    topk_min = max(0.05, topk_center - topk_range)
    topk_max = min(1.0, topk_center + topk_range)
    topk_values = np.arange(topk_min, topk_max + topk_step, topk_step).tolist()
    
    configs = list(itertools.product(alpha_values, topk_values))
    total_configs = len(configs)
    
    console.print(f"\n[bold cyan]Stage 2: Fine-Grained Search[/bold cyan]")
    console.print(f"Testing {total_configs} configurations...")
    console.print(f"  Alpha: [{alpha_min:.1f}, {alpha_max:.1f}] step {alpha_step} ({len(alpha_values)} values)")
    console.print(f"  Top-K: [{topk_min:.2f}, {topk_max:.2f}] step {topk_step} ({len(topk_values)} values)")
    console.print(f"  Beta: {beta} (fixed)")
    console.print(f"  Gamma: {gamma} (fixed)")
    console.print(f"  Ensemble: {ensemble} (fixed)\n")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing...", total=total_configs)
        
        for alpha, topk in configs:
            config = TRACERConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k_percentile=topk,
                ensemble_weight_max=ensemble
            )
            
            # Use fast evaluation if precomputed data available, otherwise fall back to legacy
            if precomputed_data is not None:
                auroc = evaluate_config_fast(config, precomputed_data)
            else:
                auroc = evaluate_config(config, merged_results)
            
            if auroc is not None:
                results.append({
                    'config': {
                        'alpha': float(alpha),
                        'beta': beta,
                        'gamma': gamma,
                        'top_k_percentile': float(topk),
                        'ensemble_weight_max': ensemble
                    },
                    'auroc': auroc
                })
            
            progress.update(task, advance=1)
    
    # Sort by AUROC
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    return {
        'mode': 'fine',
        'total_configurations': total_configs,
        'successful_evaluations': len(results),
        'best_config': results[0]['config'],
        'best_auroc': results[0]['auroc'],
        'top_10_configs': results[:10]
    }


def ensemble_optimization(
    merged_results: Results,
    console: Console,
    alpha: float = 4.0,
    beta: float = 4.0,
    gamma: float = 5.0,
    topk: float = 0.26,
    ensemble_weights: list[float] = None,
    precomputed_data: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Stage 3: Optimize ensemble weight while keeping other parameters fixed.
    
    Args:
        merged_results: Merged simulation results (used only if precomputed_data is None)
        console: Rich console for output
        alpha: Fixed alpha value
        beta: Fixed beta value
        gamma: Fixed gamma value
        topk: Fixed top_k_percentile value
        ensemble_weights: List of ensemble weights to try
        precomputed_data: Pre-computed step metrics (if None, will use legacy evaluation)
        
    Returns:
        Dictionary with best configuration and results
    """
    if ensemble_weights is None:
        ensemble_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    total_configs = len(ensemble_weights)
    
    console.print(f"\n[bold cyan]Stage 3: Ensemble Optimization[/bold cyan]")
    console.print(f"Testing {total_configs} ensemble weights...")
    console.print(f"  Alpha: {alpha} (fixed)")
    console.print(f"  Beta: {beta} (fixed)")
    console.print(f"  Gamma: {gamma} (fixed)")
    console.print(f"  Top-K: {topk} (fixed)")
    console.print(f"  Ensemble: {ensemble_weights}\n")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing...", total=total_configs)
        
        for ens in ensemble_weights:
            config = TRACERConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k_percentile=topk,
                ensemble_weight_max=ens
            )
            
            # Use fast evaluation if precomputed data available, otherwise fall back to legacy
            if precomputed_data is not None:
                auroc = evaluate_config_fast(config, precomputed_data)
            else:
                auroc = evaluate_config(config, merged_results)
            
            if auroc is not None:
                results.append({
                    'config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'top_k_percentile': topk,
                        'ensemble_weight_max': ens
                    },
                    'auroc': auroc
                })
            
            progress.update(task, advance=1)
    
    # Sort by AUROC
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    return {
        'mode': 'ensemble',
        'total_configurations': total_configs,
        'successful_evaluations': len(results),
        'best_config': results[0]['config'],
        'best_auroc': results[0]['auroc'],
        'all_results': results
    }


def custom_grid_search(
    merged_results: Results,
    console: Console,
    alpha_values: list[float],
    beta_values: list[float],
    gamma_values: list[float],
    topk_values: list[float],
    ensemble_values: list[float],
    precomputed_data: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Custom grid search with user-specified parameter values.
    
    Args:
        merged_results: Merged simulation results (used only if precomputed_data is None)
        console: Rich console for output
        alpha_values: List of alpha values to try
        beta_values: List of beta values to try
        gamma_values: List of gamma values to try
        topk_values: List of top_k_percentile values to try
        ensemble_values: List of ensemble_weight_max values to try
        precomputed_data: Pre-computed step metrics (if None, will use legacy evaluation)
        
    Returns:
        Dictionary with best configuration and results
    """
    configs = list(itertools.product(
        alpha_values, beta_values, gamma_values, topk_values, ensemble_values
    ))
    total_configs = len(configs)
    
    console.print(f"\n[bold cyan]Custom Grid Search[/bold cyan]")
    console.print(f"Testing {total_configs} configurations...")
    console.print(f"  Alpha: {alpha_values}")
    console.print(f"  Beta: {beta_values}")
    console.print(f"  Gamma: {gamma_values}")
    console.print(f"  Top-K: {topk_values}")
    console.print(f"  Ensemble: {ensemble_values}\n")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing...", total=total_configs)
        
        for alpha, beta, gamma, topk, ens in configs:
            config = TRACERConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k_percentile=topk,
                ensemble_weight_max=ens
            )
            
            # Use fast evaluation if precomputed data available, otherwise fall back to legacy
            if precomputed_data is not None:
                auroc = evaluate_config_fast(config, precomputed_data)
            else:
                auroc = evaluate_config(config, merged_results)
            
            if auroc is not None:
                results.append({
                    'config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'top_k_percentile': topk,
                        'ensemble_weight_max': ens
                    },
                    'auroc': auroc
                })
            
            progress.update(task, advance=1)
    
    # Sort by AUROC
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    return {
        'mode': 'custom',
        'total_configurations': total_configs,
        'successful_evaluations': len(results),
        'best_config': results[0]['config'],
        'best_auroc': results[0]['auroc'],
        'top_20_configs': results[:20]
    }


def print_results(optimization_results: dict[str, Any], console: Console):
    """Print optimization results in a formatted table."""
    console.print("\n" + "=" * 80)
    console.print("[bold green]OPTIMIZATION COMPLETE[/bold green]", justify="center")
    console.print("=" * 80 + "\n")
    
    console.print(f"[bold]Mode:[/bold] {optimization_results['mode']}")
    console.print(f"[bold]Total Configurations Tested:[/bold] {optimization_results['total_configurations']}")
    console.print(f"[bold]Successful Evaluations:[/bold] {optimization_results['successful_evaluations']}")
    
    # Show stage-by-stage progress for auto mode
    if optimization_results['mode'] == 'auto':
        console.print("\n[bold cyan]Stage-by-Stage Progress:[/bold cyan]")
        
        stage1 = optimization_results['stage_1_coarse']
        stage2 = optimization_results['stage_2_fine']
        stage3 = optimization_results['stage_3_ensemble']
        
        console.print(f"  Stage 1 (Coarse):   AUROC = {stage1['best_auroc']:.4f} ({stage1['configurations_tested']} configs)")
        console.print(f"  Stage 2 (Fine):     AUROC = {stage2['best_auroc']:.4f} ({stage2['configurations_tested']} configs, +{stage2['best_auroc'] - stage1['best_auroc']:.4f})")
        console.print(f"  Stage 3 (Ensemble): AUROC = {stage3['best_auroc']:.4f} ({stage3['configurations_tested']} configs, +{stage3['best_auroc'] - stage2['best_auroc']:.4f})")
        console.print(f"\n  [bold]Total Improvement:[/bold] +{optimization_results['total_improvement']:.4f} ({optimization_results['total_improvement']/optimization_results['baseline_auroc']*100:.1f}%)")
    
    # Best configuration
    console.print("\n[bold cyan]Best Configuration:[/bold cyan]")
    best_config = optimization_results['best_config']
    console.print(f"  Alpha (α): {best_config['alpha']}")
    console.print(f"  Beta (β): {best_config['beta']}")
    console.print(f"  Gamma (γ): {best_config['gamma']}")
    console.print(f"  Top-K Percentile: {best_config['top_k_percentile']}")
    console.print(f"  Ensemble Weight (Max): {best_config['ensemble_weight_max']}")
    console.print(f"\n  [bold green]AUROC: {optimization_results['best_auroc']:.4f}[/bold green]")
    
    # Top configurations table (skip for auto mode as it's redundant)
    if optimization_results['mode'] != 'auto':
        if 'top_10_configs' in optimization_results:
            configs_to_show = optimization_results['top_10_configs'][:10]
        elif 'top_20_configs' in optimization_results:
            configs_to_show = optimization_results['top_20_configs'][:10]
        elif 'all_results' in optimization_results:
            configs_to_show = optimization_results['all_results'][:10]
        else:
            configs_to_show = []
        
        if configs_to_show:
            console.print(f"\n[bold cyan]Top {len(configs_to_show)} Configurations:[/bold cyan]\n")
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Rank", justify="right", width=5)
            table.add_column("AUROC", justify="right", width=8)
            table.add_column("Alpha", justify="right", width=7)
            table.add_column("Beta", justify="right", width=7)
            table.add_column("Gamma", justify="right", width=7)
            table.add_column("Top-K", justify="right", width=8)
            table.add_column("Ensemble", justify="right", width=9)
            
            for i, result in enumerate(configs_to_show, 1):
                cfg = result['config']
                auroc = result['auroc']
                
                # Color code by performance
                if auroc >= 0.7:
                    auroc_str = f"[green]{auroc:.4f}[/green]"
                elif auroc >= 0.65:
                    auroc_str = f"[yellow]{auroc:.4f}[/yellow]"
                else:
                    auroc_str = f"{auroc:.4f}"
                
                table.add_row(
                    str(i),
                    auroc_str,
                    f"{cfg['alpha']:.2f}",
                    f"{cfg['beta']:.2f}",
                    f"{cfg['gamma']:.2f}",
                    f"{cfg['top_k_percentile']:.2f}",
                    f"{cfg['ensemble_weight_max']:.2f}"
                )
            
            console.print(table)
    
    console.print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize TRACER parameters for failure prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "simulation_path",
        type=str,
        help="Path to simulation JSON file or directory containing JSON files"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "coarse", "fine", "ensemble", "custom", "random", "wide"],
        default="auto",
        help="Optimization mode: auto (3 stages), random (random search), wide (broad grid + random)"
    )
    
    # Random search parameters
    parser.add_argument(
        "--n-random",
        type=int,
        default=500,
        help="Number of random configurations to test in random search mode (default: 500)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    # Fine-grained search parameters
    parser.add_argument("--alpha-center", type=float, default=7.0)
    parser.add_argument("--alpha-range", type=float, default=3.0)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--topk-center", type=float, default=0.25)
    parser.add_argument("--topk-range", type=float, default=0.1)
    parser.add_argument("--topk-step", type=float, default=0.01)
    
    # Fixed parameters for fine-grained and ensemble modes
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--top-k", type=float, default=0.26)
    
    # Ensemble weights
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="+",
        default=None,
        help="List of ensemble weights to try (e.g., 0.1 0.2 0.3)"
    )
    
    # Custom grid search parameters
    parser.add_argument("--alpha-values", type=float, nargs="+", default=None)
    parser.add_argument("--beta-values", type=float, nargs="+", default=None)
    parser.add_argument("--gamma-values", type=float, nargs="+", default=None)
    parser.add_argument("--topk-values", type=float, nargs="+", default=None)
    parser.add_argument("--ensemble-values", type=float, nargs="+", default=None)
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: auto-saves to data/optimization/ with same name as input)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file, only print to console"
    )
    
    args = parser.parse_args()
    
    # Setup
    console = Console()
    sim_path = Path(args.simulation_path)
    
    if not sim_path.exists():
        logger.error(f"Path not found: {sim_path}")
        sys.exit(1)
    
    # Load and merge simulations
    console.print("[bold]Loading simulation file(s)...[/bold]")
    results_list = load_simulations_from_path(sim_path)
    merged_results = merge_results(results_list)
    
    total_sims = len(merged_results.simulations)
    console.print(f"[green]✓ Loaded {total_sims} simulations total[/green]\n")
    
    # Pre-compute step-level metrics (Phase 1) - run once for all parameter searches
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]PHASE 1: PRE-COMPUTING STEP METRICS[/bold magenta]", justify="center")
    console.print("=" * 80 + "\n")
    
    precomputed_data = precompute_step_metrics(merged_results, console)
    
    # Run optimization based on mode
    if args.mode == "auto":
        # Stage 1: Coarse grid search
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/3: COARSE GRID SEARCH[/bold magenta]", justify="center")
        console.print("=" * 80 + "\n")
        
        coarse_results = coarse_grid_search(merged_results, console, precomputed_data)
        best_coarse = coarse_results['best_config']
        
        console.print(f"\n[green]✓ Stage 1 Complete: AUROC = {coarse_results['best_auroc']:.4f}[/green]")
        console.print(f"  Best coarse params: α={best_coarse['alpha']}, β={best_coarse['beta']}, γ={best_coarse['gamma']}, k={best_coarse['top_k_percentile']}\n")
        
        # Stage 2: Fine-grained search around best coarse params
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/3: FINE-GRAINED SEARCH[/bold magenta]", justify="center")
        console.print("=" * 80 + "\n")
        
        fine_results = fine_grained_search(
            merged_results,
            console,
            alpha_center=best_coarse['alpha'],
            alpha_range=3.0,
            alpha_step=0.5,
            topk_center=best_coarse['top_k_percentile'],
            topk_range=0.1,
            topk_step=0.01,
            beta=best_coarse['beta'],
            gamma=best_coarse['gamma'],
            ensemble=best_coarse['ensemble_weight_max'],
            precomputed_data=precomputed_data
        )
        best_fine = fine_results['best_config']
        
        console.print(f"\n[green]✓ Stage 2 Complete: AUROC = {fine_results['best_auroc']:.4f}[/green]")
        console.print(f"  Improvement: +{fine_results['best_auroc'] - coarse_results['best_auroc']:.4f}")
        console.print(f"  Best fine params: α={best_fine['alpha']}, k={best_fine['top_k_percentile']}\n")
        
        # Stage 3: Ensemble optimization
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 3/3: ENSEMBLE OPTIMIZATION[/bold magenta]", justify="center")
        console.print("=" * 80 + "\n")
        
        ensemble_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        ensemble_results = ensemble_optimization(
            merged_results,
            console,
            alpha=best_fine['alpha'],
            beta=best_fine['beta'],
            gamma=best_fine['gamma'],
            topk=best_fine['top_k_percentile'],
            ensemble_weights=ensemble_weights,
            precomputed_data=precomputed_data
        )
        
        console.print(f"\n[green]✓ Stage 3 Complete: AUROC = {ensemble_results['best_auroc']:.4f}[/green]")
        console.print(f"  Improvement: +{ensemble_results['best_auroc'] - fine_results['best_auroc']:.4f}")
        console.print(f"  Best ensemble weight: {ensemble_results['best_config']['ensemble_weight_max']}\n")
        
        # Combine all results
        optimization_results = {
            'mode': 'auto',
            'total_configurations': (
                coarse_results['total_configurations'] +
                fine_results['total_configurations'] +
                ensemble_results['total_configurations']
            ),
            'successful_evaluations': (
                coarse_results['successful_evaluations'] +
                fine_results['successful_evaluations'] +
                ensemble_results['successful_evaluations']
            ),
            'best_config': ensemble_results['best_config'],
            'best_auroc': ensemble_results['best_auroc'],
            'stage_1_coarse': {
                'best_config': coarse_results['best_config'],
                'best_auroc': coarse_results['best_auroc'],
                'configurations_tested': coarse_results['total_configurations']
            },
            'stage_2_fine': {
                'best_config': fine_results['best_config'],
                'best_auroc': fine_results['best_auroc'],
                'configurations_tested': fine_results['total_configurations']
            },
            'stage_3_ensemble': {
                'best_config': ensemble_results['best_config'],
                'best_auroc': ensemble_results['best_auroc'],
                'configurations_tested': ensemble_results['total_configurations']
            },
            'baseline_auroc': coarse_results['best_auroc'],
            'total_improvement': ensemble_results['best_auroc'] - coarse_results['best_auroc']
        }
    
    elif args.mode == "coarse":
        optimization_results = coarse_grid_search(merged_results, console, precomputed_data)
    
    elif args.mode == "random":
        # Random search mode - explore parameter space randomly
        optimization_results = random_search(
            merged_results,
            console,
            n_iterations=args.n_random,
            alpha_range=(0.0, 20.0),
            beta_range=(0.0, 20.0),
            gamma_range=(0.0, 20.0),
            topk_range=(0.05, 1.0),
            ensemble_range=(0.0, 0.5),
            precomputed_data=precomputed_data,
            seed=args.random_seed
        )
    
    elif args.mode == "wide":
        # Wide search: combination of broad grid + random exploration
        console.print("\n[bold magenta]WIDE SEARCH MODE: Grid + Random Exploration[/bold magenta]\n")
        
        # Stage 1: Very wide grid search
        console.print("[bold]Phase 1: Wide Grid Search[/bold]")
        wide_alpha = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
        wide_beta = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
        wide_gamma = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
        wide_topk = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        wide_ensemble = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        wide_configs = list(itertools.product(wide_alpha[:8], wide_beta[:5], wide_gamma[:5], wide_topk[:6], wide_ensemble[:3]))
        total_wide = len(wide_configs)
        
        console.print(f"Testing {total_wide} wide grid configurations...\n")
        
        wide_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_wide = progress.add_task("Wide Grid...", total=total_wide)
            
            for alpha, beta, gamma, topk, ensemble in wide_configs:
                config = TRACERConfig(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    top_k_percentile=topk,
                    ensemble_weight_max=ensemble
                )
                
                if precomputed_data is not None:
                    auroc = evaluate_config_fast(config, precomputed_data)
                else:
                    auroc = evaluate_config(config, merged_results)
                
                if auroc is not None:
                    wide_results.append({
                        'config': {
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                            'top_k_percentile': topk,
                            'ensemble_weight_max': ensemble
                        },
                        'auroc': auroc
                    })
                
                progress.update(task_wide, advance=1)
        
        wide_results.sort(key=lambda x: x['auroc'], reverse=True)
        console.print(f"\n✓ Wide Grid Complete: Best AUROC = {wide_results[0]['auroc']:.4f}\n")
        
        # Stage 2: Random search with very wide bounds
        console.print("[bold]Phase 2: Random Exploration[/bold]")
        random_results = random_search(
            merged_results,
            console,
            n_iterations=1000,
            alpha_range=(0.0, 30.0),
            beta_range=(0.0, 30.0),
            gamma_range=(0.0, 30.0),
            topk_range=(0.01, 1.0),
            ensemble_range=(0.0, 0.6),
            precomputed_data=precomputed_data,
            seed=args.random_seed
        )
        
        # Combine results
        all_results = wide_results + random_results.get('top_10_configs', [])
        all_results.sort(key=lambda x: x['auroc'], reverse=True)
        
        optimization_results = {
            'mode': 'wide',
            'total_configurations': total_wide + args.n_random,
            'successful_evaluations': len(wide_results) + random_results['successful_evaluations'],
            'best_config': all_results[0]['config'],
            'best_auroc': all_results[0]['auroc'],
            'top_10_configs': all_results[:10],
            'wide_grid_best': wide_results[0]['auroc'],
            'random_search_best': random_results['best_auroc']
        }

    elif args.mode == "fine":
        optimization_results = fine_grained_search(
            merged_results,
            console,
            alpha_center=args.alpha_center,
            alpha_range=args.alpha_range,
            alpha_step=args.alpha_step,
            topk_center=args.topk_center,
            topk_range=args.topk_range,
            topk_step=args.topk_step,
            beta=args.beta,
            gamma=args.gamma,
            ensemble=args.ensemble_weights[0] if args.ensemble_weights else 0.15,
            precomputed_data=precomputed_data
        )
    
    elif args.mode == "ensemble":
        optimization_results = ensemble_optimization(
            merged_results,
            console,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            topk=args.top_k,
            ensemble_weights=args.ensemble_weights,
            precomputed_data=precomputed_data
        )
    
    elif args.mode == "custom":
        # Validate custom parameters
        if not all([
            args.alpha_values,
            args.beta_values,
            args.gamma_values,
            args.topk_values,
            args.ensemble_values
        ]):
            logger.error("Custom mode requires --alpha-values, --beta-values, --gamma-values, --topk-values, and --ensemble-values")
            sys.exit(1)
        
        optimization_results = custom_grid_search(
            merged_results,
            console,
            alpha_values=args.alpha_values,
            beta_values=args.beta_values,
            gamma_values=args.gamma_values,
            topk_values=args.topk_values,
            ensemble_values=args.ensemble_values,
            precomputed_data=precomputed_data
        )
    
    # Add metadata
    optimization_results['timestamp'] = datetime.now().isoformat()
    optimization_results['simulation_path'] = str(sim_path)
    optimization_results['total_simulations'] = total_sims
    
    # Print results
    print_results(optimization_results, console)
    
    # Determine output path
    if not args.no_save:
        if args.output:
            # User specified custom output path
            output_path = Path(args.output)
        else:
            # Auto-generate output path in data/optimization/
            # Get the base project directory
            project_root = Path(__file__).parent.parent.parent.parent
            optimization_dir = project_root / "data" / "optimization"
            optimization_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the same filename as input (or generate one for directories)
            if sim_path.is_file():
                output_filename = sim_path.name
            else:
                # For directories, use a timestamp-based name
                output_filename = f"optimization_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.json"
            
            output_path = optimization_dir / output_filename
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        console.print(f"\n[green]✓ Results saved to {output_path}[/green]")
    
    # Print usage recommendation
    console.print("\n[bold cyan]To use the best configuration:[/bold cyan]")
    best = optimization_results['best_config']
    console.print(f"python -m tau2.scripts.analyze_uncertainty data/simulations/file.json \\")
    console.print(f"  --tracer-config '{{")
    console.print(f"    \"alpha\": {best['alpha']},")
    console.print(f"    \"beta\": {best['beta']},")
    console.print(f"    \"gamma\": {best['gamma']},")
    console.print(f"    \"top_k_percentile\": {best['top_k_percentile']},")
    console.print(f"    \"ensemble_weight_max\": {best['ensemble_weight_max']}")
    console.print(f"  }}'\n")


if __name__ == "__main__":
    main()

