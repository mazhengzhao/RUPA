#!/usr/bin/env python3
"""
Quick test script to verify the optimized TRACER parameter search works correctly.

This script runs a small-scale optimization to validate:
1. Pre-computation phase completes successfully
2. Fast evaluation produces identical AUROC results as legacy evaluation
3. Performance improvement is significant
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from tau2.data_model.simulation import Results
from tau2.metrics.uncertainty import TRACERConfig
from tau2.scripts.optimize_tracer_parameters import (
    precompute_step_metrics,
    evaluate_config_fast,
    evaluate_config,
    load_simulations_from_path,
    merge_results
)

def test_optimization(sim_path: Path):
    """Test the optimized parameter search on a real dataset."""
    console = Console()
    
    console.print("[bold cyan]Testing Optimized TRACER Parameter Search[/bold cyan]\n")
    
    # Load data
    console.print("1. Loading simulation data...")
    results_list = load_simulations_from_path(sim_path)
    merged_results = merge_results(results_list)
    total_sims = len(merged_results.simulations)
    console.print(f"   ✓ Loaded {total_sims} simulations\n")
    
    # Test configurations (small sample)
    test_configs = [
        TRACERConfig(alpha=3.0, beta=4.0, gamma=5.0, top_k_percentile=0.25, ensemble_weight_max=0.15),
        TRACERConfig(alpha=4.0, beta=4.0, gamma=5.0, top_k_percentile=0.26, ensemble_weight_max=0.2),
        TRACERConfig(alpha=5.0, beta=5.0, gamma=6.0, top_k_percentile=0.3, ensemble_weight_max=0.15),
    ]
    
    # Pre-computation phase
    console.print("2. Testing pre-computation phase...")
    start_precomp = time.time()
    precomputed_data = precompute_step_metrics(merged_results, console)
    precomp_time = time.time() - start_precomp
    console.print(f"   ✓ Pre-computation completed in {precomp_time:.2f}s\n")
    
    # Test legacy evaluation (slower)
    console.print("3. Testing legacy evaluation (baseline)...")
    legacy_results = []
    start_legacy = time.time()
    for i, config in enumerate(test_configs, 1):
        auroc = evaluate_config(config, merged_results, verbose=False)
        legacy_results.append(auroc)
        console.print(f"   Config {i}: AUROC = {auroc:.4f if auroc else 'N/A'}")
    legacy_time = time.time() - start_legacy
    console.print(f"   Total time: {legacy_time:.2f}s\n")
    
    # Test fast evaluation (optimized)
    console.print("4. Testing fast evaluation (optimized)...")
    fast_results = []
    start_fast = time.time()
    for i, config in enumerate(test_configs, 1):
        auroc = evaluate_config_fast(config, precomputed_data, verbose=False)
        fast_results.append(auroc)
        console.print(f"   Config {i}: AUROC = {auroc:.4f if auroc else 'N/A'}")
    fast_time = time.time() - start_fast
    console.print(f"   Total time: {fast_time:.2f}s\n")
    
    # Validation
    console.print("5. Validation...")
    all_match = True
    for i, (legacy, fast) in enumerate(zip(legacy_results, fast_results), 1):
        if legacy is None or fast is None:
            console.print(f"   [yellow]⚠️  Config {i}: One or both evaluations failed[/yellow]")
            continue
        
        diff = abs(legacy - fast)
        if diff < 0.0001:
            console.print(f"   [green]✓ Config {i}: Results match (diff = {diff:.6f})[/green]")
        else:
            console.print(f"   [red]✗ Config {i}: Results differ! (diff = {diff:.6f})[/red]")
            all_match = False
    
    console.print()
    
    # Performance summary
    console.print("[bold cyan]Performance Summary[/bold cyan]")
    console.print(f"  Pre-computation time: {precomp_time:.2f}s")
    console.print(f"  Legacy evaluation time: {legacy_time:.2f}s ({legacy_time/len(test_configs):.2f}s per config)")
    console.print(f"  Fast evaluation time: {fast_time:.2f}s ({fast_time/len(test_configs):.2f}s per config)")
    
    if fast_time > 0:
        speedup = legacy_time / fast_time
        console.print(f"\n  [bold green]Speedup: {speedup:.1f}×[/bold green]")
        
        # Project speedup for full optimization
        full_configs = 1125 + 315 + 9  # Typical auto mode (coarse + fine + ensemble)
        projected_legacy = (legacy_time / len(test_configs)) * full_configs
        projected_fast = precomp_time + (fast_time / len(test_configs)) * full_configs
        
        console.print(f"\n  [bold]Projected for full optimization ({full_configs} configs):[/bold]")
        console.print(f"    Legacy approach: ~{projected_legacy/60:.1f} minutes")
        console.print(f"    Optimized approach: ~{projected_fast/60:.1f} minutes")
        console.print(f"    Time saved: ~{(projected_legacy - projected_fast)/60:.1f} minutes")
    
    # Final verdict
    console.print()
    if all_match:
        console.print("[bold green]✅ SUCCESS: All results match! Optimization is correct.[/bold green]")
        return True
    else:
        console.print("[bold red]❌ FAILURE: Some results don't match. Investigation needed.[/bold red]")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_optimization.py <path_to_simulation_file.json>")
        print("\nExample:")
        print("  python test_optimization.py data/simulations/2026-01-02T15:21:31.412260_airline_llm_agent_gemini-2.5-flash_user_simulator_gemini-2.5-flash.json")
        sys.exit(1)
    
    sim_path = Path(sys.argv[1])
    if not sim_path.exists():
        print(f"Error: File not found: {sim_path}")
        sys.exit(1)
    
    success = test_optimization(sim_path)
    sys.exit(0 if success else 1)

