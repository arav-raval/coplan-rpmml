"""
Comparison Experiment: Communication ON vs OFF

Compares baseline (no communication) with optimal communication policy
to demonstrate the value of coordination.
"""

import time
import numpy as np
from src.simulation import run_batch_multiple_seeds
from src.visualization.plots import plot_comparison
from src.config import (
    EXPERIMENT_CONFIG, NUM_SEEDS, NUM_SEEDS_QUICK, SIMULATION_FPS,
    OPTIMAL_COMM_POLICY
)


def run_comparison_sweep(quick=False, visualize=False):
    """
    Compare communication ON (with optimal policy) vs OFF (baseline).
    
    Args:
        quick: If True, use fewer seeds
        visualize: If True, show visualization (not implemented for comparison)
    
    Returns:
        dict: Results from both scenarios
    """
    print("\n" + "=" * 75)
    print("COMPARISON EXPERIMENT: Communication ON vs OFF")
    print("=" * 75)
    
    # Get configuration
    n_agents = EXPERIMENT_CONFIG['n_agents']
    comm_range = EXPERIMENT_CONFIG['comm_range']
    num_seeds = NUM_SEEDS_QUICK if quick else NUM_SEEDS
    
    # Get optimal policy from config
    opt_freq_steps = OPTIMAL_COMM_POLICY['frequency_steps']
    opt_msg_length = OPTIMAL_COMM_POLICY['msg_length']
    opt_freq_hz = SIMULATION_FPS / opt_freq_steps
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Comm range: {comm_range:.0f}px")
    print(f"\nOptimal Policy (from 2D sweep):")
    print(f"  Frequency: {opt_freq_steps} steps ({opt_freq_hz:.2f} Hz)")
    print(f"  Message length: {opt_msg_length} waypoints")
    print(f"  Description: {OPTIMAL_COMM_POLICY['description']}")
    print()
    
    start_time = time.time()
    
    # Scenario 1: No Communication (Baseline)
    print("=" * 75)
    print("SCENARIO 1: NO COMMUNICATION (Baseline)")
    print("=" * 75)
    print("Running simulations with comm_enabled=False...")
    print()
    
    result_no_comm = run_batch_multiple_seeds(
        comm_enabled=False,
        broadcast_interval_steps=999,  # Irrelevant when comm disabled
        comm_range=comm_range,
        msg_length=1,  # Irrelevant when comm disabled
        n_agents=n_agents,
        num_seeds=num_seeds,
        verbose=False
    )
    
    print(f"✓ Completed baseline (no communication)")
    print(f"  Cost: {result_no_comm['cost_mean']:.1f} ± {result_no_comm['cost_std']:.1f}")
    print(f"  Collisions: {result_no_comm['collision_count_mean']:.2f} ± {result_no_comm['collision_count_std']:.2f}")
    print(f"  Collision Rate: {result_no_comm['collision_rate']*100:.1f}%")
    print()
    
    # Scenario 2: Optimal Communication Policy
    print("=" * 75)
    print("SCENARIO 2: OPTIMAL COMMUNICATION POLICY")
    print("=" * 75)
    print(f"Running simulations with optimal policy...")
    print(f"  Frequency: {opt_freq_steps} steps ({opt_freq_hz:.2f} Hz)")
    print(f"  Message length: {opt_msg_length} waypoints")
    print()
    
    result_with_comm = run_batch_multiple_seeds(
        comm_enabled=True,
        broadcast_interval_steps=opt_freq_steps,
        comm_range=comm_range,
        msg_length=opt_msg_length,
        n_agents=n_agents,
        num_seeds=num_seeds,
        verbose=False
    )
    
    print(f"✓ Completed optimal policy")
    print(f"  Cost: {result_with_comm['cost_mean']:.1f} ± {result_with_comm['cost_std']:.1f}")
    print(f"  Collisions: {result_with_comm['collision_count_mean']:.2f} ± {result_with_comm['collision_count_std']:.2f}")
    print(f"  Collision Rate: {result_with_comm['collision_rate']*100:.1f}%")
    print()
    
    elapsed = time.time() - start_time
    
    # Calculate improvements
    cost_improvement = ((result_no_comm['cost_mean'] - result_with_comm['cost_mean']) / 
                       result_no_comm['cost_mean'] * 100)
    collision_reduction = ((result_no_comm['collision_count_mean'] - result_with_comm['collision_count_mean']) / 
                          max(result_no_comm['collision_count_mean'], 0.01) * 100)
    
    print("=" * 75)
    print("COMPARISON SUMMARY")
    print("=" * 75)
    print(f"\n{'Metric':<30s} {'No Comm':<15s} {'Optimal':<15s} {'Change':<15s}")
    print("-" * 75)
    
    # Cost
    print(f"{'Total Cost':<30s} {result_no_comm['cost_mean']:>14.1f} "
          f"{result_with_comm['cost_mean']:>14.1f} {cost_improvement:>13.1f}%")
    
    # Time
    time_change = ((result_with_comm['time_mean'] - result_no_comm['time_mean']) / 
                   result_no_comm['time_mean'] * 100)
    print(f"{'Simulation Time (steps)':<30s} {result_no_comm['time_mean']:>14.1f} "
          f"{result_with_comm['time_mean']:>14.1f} {time_change:>13.1f}%")
    
    # Replans
    print(f"{'Replanning Events':<30s} {result_no_comm['replan_mean']:>14.1f} "
          f"{result_with_comm['replan_mean']:>14.1f} "
          f"{result_with_comm['replan_mean']:>14.1f}")
    
    # Messages
    messages_per_run = result_with_comm.get('messages_sent_mean', result_with_comm['replan_mean'])
    print(f"{'Messages Sent':<30s} {'0':>14s} "
          f"{messages_per_run:>14.1f} {messages_per_run:>14.1f}")
    
    # Safety
    sep_improvement = ((result_with_comm['avg_separation_mean'] - result_no_comm['avg_separation_mean']) / 
                      max(result_no_comm['avg_separation_mean'], 1) * 100)
    print(f"{'Avg Separation (px)':<30s} {result_no_comm['avg_separation_mean']:>14.1f} "
          f"{result_with_comm['avg_separation_mean']:>14.1f} {sep_improvement:>13.1f}%")
    
    # Collisions
    print(f"{'Collision Count':<30s} {result_no_comm['collision_count_mean']:>14.2f} "
          f"{result_with_comm['collision_count_mean']:>14.2f} {collision_reduction:>13.1f}%")
    
    print(f"{'Collision Rate':<30s} {result_no_comm['collision_rate']*100:>13.1f}% "
          f"{result_with_comm['collision_rate']*100:>13.1f}% "
          f"{(result_with_comm['collision_rate'] - result_no_comm['collision_rate'])*100:>13.1f}%")
    
    print()
    print(f"✓ Comparison completed in {elapsed:.1f}s")
    
    # Generate plot
    print("\nGenerating comparison plot...")
    config = {
        'n_agents': n_agents,
        'num_seeds': num_seeds,
        'comm_range': comm_range,
        'opt_freq_steps': opt_freq_steps,
        'opt_msg_length': opt_msg_length,
    }
    
    plot_comparison(
        result_no_comm,
        result_with_comm,
        save_path="results/comparison.png",
        config=config
    )
    
    print(f"✓ Saved to results/comparison.png")
    
    return {
        'no_comm': result_no_comm,
        'with_comm': result_with_comm,
        'improvement': {
            'cost_pct': cost_improvement,
            'collision_reduction_pct': collision_reduction,
            'separation_improvement_pct': sep_improvement
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare communication ON vs OFF")
    parser.add_argument('--quick', action='store_true', help='Use fewer seeds')
    args = parser.parse_args()
    
    run_comparison_sweep(quick=args.quick)
