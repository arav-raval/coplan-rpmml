"""
Master experiment runner for all parameter sweeps.

Usage:
    python experiments/run_all.py                    # Run all experiments
    python experiments/run_all.py --list             # List available experiments
    python experiments/run_all.py frequency          # Run specific experiment
    python experiments/run_all.py frequency range    # Run multiple experiments
    python experiments/run_all.py --quick            # Quick mode (fewer trials)
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from src.simulation import run_batch
from src.visualization.plots import (
    plot_frequency_sweep, 
    plot_range_sweep, 
    plot_cost_landscape, 
    plot_agent_sweep,
    identify_plateaus
)


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def run_frequency_sweep(quick=False):
    """Sweep over broadcast frequency."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Broadcast Frequency Sweep")
    print("=" * 60)
    
    # Parameters
    frequencies = np.linspace(0.2, 5.0, 8 if quick else 15)
    comm_range = 250.0
    msg_length = 10
    n_agents = 4
    num_trials = 5 if quick else 10
    
    print(f"Frequencies: {frequencies[0]:.2f} to {frequencies[-1]:.2f} Hz ({len(frequencies)} values)")
    print(f"Config: {n_agents} agents, {num_trials} trials, range={comm_range}px\n")
    
    results = []
    for i, freq in enumerate(frequencies):
        print(f"[{i+1:2d}/{len(frequencies)}] freq={freq:.2f} Hz ...", end=" ", flush=True)
        result = run_batch(
            comm_enabled=True,
            broadcast_frequency=freq,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            verbose=False
        )
        print(f"cost={result['cost_mean']:6.1f}, safety={result['min_separation_mean']:5.1f}px")
        result['frequency'] = freq
        results.append(result)
    
    config = {'n_agents': n_agents, 'num_trials': num_trials, 
              'comm_range': comm_range, 'msg_length': msg_length}
    plot_frequency_sweep(frequencies, results, config=config)
    
    # Return summary
    costs = [r['cost_mean'] for r in results]
    opt_idx = np.argmin(costs)
    return {
        'optimal_frequency': frequencies[opt_idx],
        'optimal_cost': costs[opt_idx],
        'results': results
    }


def run_range_sweep(quick=False):
    """Sweep over communication range."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Communication Range Sweep")
    print("=" * 60)
    
    ranges = np.linspace(50, 400, 8 if quick else 15)
    frequency = 1.5
    msg_length = 10
    n_agents = 4
    num_trials = 5 if quick else 10
    
    print(f"Ranges: {ranges[0]:.0f} to {ranges[-1]:.0f} px ({len(ranges)} values)")
    print(f"Config: {n_agents} agents, {num_trials} trials, freq={frequency}Hz\n")
    
    results = []
    for i, r in enumerate(ranges):
        print(f"[{i+1:2d}/{len(ranges)}] range={r:3.0f}px ...", end=" ", flush=True)
        result = run_batch(
            comm_enabled=True,
            broadcast_frequency=frequency,
            comm_range=r,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            verbose=False
        )
        print(f"cost={result['cost_mean']:6.1f}, collisions={result['collision_rate']:.0%}")
        result['range'] = r
        results.append(result)
    
    config = {'n_agents': n_agents, 'num_trials': num_trials,
              'frequency': frequency, 'msg_length': msg_length}
    plot_range_sweep(ranges, results, config=config)
    
    costs = [r['cost_mean'] for r in results]
    opt_idx = np.argmin(costs)
    return {
        'optimal_range': ranges[opt_idx],
        'optimal_cost': costs[opt_idx],
        'results': results
    }


def run_agent_sweep(quick=False):
    """Sweep over number of agents."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Agent Count Sweep")
    print("=" * 60)
    
    agent_counts = [2, 3, 4, 5, 6] if quick else [2, 3, 4, 5, 6, 7, 8]
    frequency = 1.5
    comm_range = 250.0
    msg_length = 10
    num_trials = 5 if quick else 10
    
    print(f"Agent counts: {agent_counts}")
    print(f"Config: freq={frequency}Hz, range={comm_range}px, {num_trials} trials\n")
    
    results = []
    for i, n_agents in enumerate(agent_counts):
        print(f"[{i+1}/{len(agent_counts)}] agents={n_agents} ...", end=" ", flush=True)
        result = run_batch(
            comm_enabled=True,
            broadcast_frequency=frequency,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            verbose=False
        )
        print(f"cost={result['cost_mean']:6.1f}, collisions={result['collision_rate']:.0%}")
        result['n_agents'] = n_agents
        results.append(result)
    
    config = {'num_trials': num_trials, 'frequency': frequency,
              'comm_range': comm_range, 'msg_length': msg_length}
    plot_agent_sweep(agent_counts, results, config=config)
    
    collision_rates = [r['collision_rate'] for r in results]
    safe_counts = [agent_counts[i] for i, cr in enumerate(collision_rates) if cr < 0.1]
    return {
        'max_safe_agents': max(safe_counts) if safe_counts else 0,
        'results': results
    }


def run_2d_landscape(quick=False):
    """2D sweep over frequency and range."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: 2D Cost Landscape (Frequency × Range)")
    print("=" * 60)
    
    frequencies = np.linspace(0.5, 3.0, 5 if quick else 8)
    ranges = np.linspace(100, 350, 5 if quick else 8)
    msg_length = 10
    n_agents = 4
    num_trials = 3 if quick else 5
    
    total = len(frequencies) * len(ranges)
    print(f"Grid: {len(frequencies)}×{len(ranges)} = {total} combinations")
    print(f"Config: {n_agents} agents, {num_trials} trials per combo\n")
    
    cost_matrix = np.zeros((len(frequencies), len(ranges)))
    count = 0
    
    for i, freq in enumerate(frequencies):
        for j, r in enumerate(ranges):
            count += 1
            print(f"[{count:3d}/{total}] f={freq:.2f}Hz, r={r:3.0f}px ...", end=" ", flush=True)
            result = run_batch(
                comm_enabled=True,
                broadcast_frequency=freq,
                comm_range=r,
                msg_length=msg_length,
                n_agents=n_agents,
                num_trials=num_trials,
                verbose=False
            )
            cost_matrix[i, j] = result['cost_mean']
            print(f"cost={result['cost_mean']:6.1f}")
    
    config = {'n_agents': n_agents, 'num_trials': num_trials, 'msg_length': msg_length}
    plot_cost_landscape(frequencies, ranges, cost_matrix, config=config)
    
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    return {
        'optimal_frequency': frequencies[min_idx[0]],
        'optimal_range': ranges[min_idx[1]],
        'optimal_cost': cost_matrix[min_idx],
        'cost_matrix': cost_matrix
    }


def run_comm_comparison(quick=False):
    """Compare communication ON vs OFF."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Communication vs No Communication")
    print("=" * 60)
    
    n_agents = 4
    num_trials = 10 if not quick else 5
    frequency = 1.5
    comm_range = 250.0
    msg_length = 10
    
    print(f"Config: {n_agents} agents, {num_trials} trials\n")
    
    # With communication
    print("Running WITH communication...", end=" ", flush=True)
    result_comm = run_batch(
        comm_enabled=True,
        broadcast_frequency=frequency,
        comm_range=comm_range,
        msg_length=msg_length,
        n_agents=n_agents,
        num_trials=num_trials,
        verbose=False
    )
    print(f"cost={result_comm['cost_mean']:.1f}, collisions={result_comm['collision_rate']:.0%}")
    
    # Without communication
    print("Running WITHOUT communication...", end=" ", flush=True)
    result_no_comm = run_batch(
        comm_enabled=False,
        broadcast_frequency=frequency,
        comm_range=comm_range,
        msg_length=msg_length,
        n_agents=n_agents,
        num_trials=num_trials,
        verbose=False
    )
    print(f"cost={result_no_comm['cost_mean']:.1f}, collisions={result_no_comm['collision_rate']:.0%}")
    
    # Summary
    print("\n" + "-" * 40)
    print(f"{'Metric':<20} {'Comm ON':<15} {'Comm OFF':<15} {'Δ'}")
    print("-" * 40)
    print(f"{'Cost':<20} {result_comm['cost_mean']:<15.1f} {result_no_comm['cost_mean']:<15.1f} "
          f"{result_comm['cost_mean'] - result_no_comm['cost_mean']:+.1f}")
    print(f"{'Collision Rate':<20} {result_comm['collision_rate']:<15.0%} {result_no_comm['collision_rate']:<15.0%}")
    print(f"{'Min Separation':<20} {result_comm['min_separation_mean']:<15.1f} {result_no_comm['min_separation_mean']:<15.1f}")
    print(f"{'Replans':<20} {result_comm['replan_mean']:<15.1f} {result_no_comm['replan_mean']:<15.1f}")
    
    return {
        'comm_on': result_comm,
        'comm_off': result_no_comm,
        'cost_reduction': result_no_comm['cost_mean'] - result_comm['cost_mean'],
        'collision_reduction': result_no_comm['collision_rate'] - result_comm['collision_rate']
    }


# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

EXPERIMENTS = {
    'frequency': {
        'name': 'Broadcast Frequency Sweep',
        'description': 'Test effect of communication frequency on performance',
        'function': run_frequency_sweep,
    },
    'range': {
        'name': 'Communication Range Sweep', 
        'description': 'Test effect of communication range on performance',
        'function': run_range_sweep,
    },
    'agents': {
        'name': 'Agent Count Sweep',
        'description': 'Test how performance scales with number of agents',
        'function': run_agent_sweep,
    },
    '2d': {
        'name': '2D Cost Landscape',
        'description': 'Generate heatmap of cost over frequency × range',
        'function': run_2d_landscape,
    },
    'comparison': {
        'name': 'Communication Comparison',
        'description': 'Compare communication ON vs OFF',
        'function': run_comm_comparison,
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_all.py                    # Run all experiments
  python experiments/run_all.py --list             # List available experiments
  python experiments/run_all.py frequency          # Run frequency sweep only
  python experiments/run_all.py frequency range    # Run multiple experiments
  python experiments/run_all.py --quick            # Quick mode (fewer trials)
  python experiments/run_all.py --quick frequency  # Quick frequency sweep
        """
    )
    
    parser.add_argument('experiments', nargs='*', help='Experiments to run (default: all)')
    parser.add_argument('--list', '-l', action='store_true', help='List available experiments')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode (fewer trials/values)')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 60)
        for key, exp in EXPERIMENTS.items():
            print(f"  {key:<12} - {exp['name']}")
            print(f"               {exp['description']}")
        print()
        return
    
    # Determine which experiments to run
    if args.experiments:
        to_run = []
        for name in args.experiments:
            if name in EXPERIMENTS:
                to_run.append(name)
            else:
                print(f"Unknown experiment: {name}")
                print(f"   Available: {', '.join(EXPERIMENTS.keys())}")
                return
    else:
        to_run = list(EXPERIMENTS.keys())
    
    # Header
    print("\n" + "=" * 60)
    print("COPLAN EXPERIMENT SUITE")
    print("=" * 60)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Experiments: {', '.join(to_run)}")
    print("=" * 60)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for exp_name in to_run:
        exp = EXPERIMENTS[exp_name]
        start = time.time()
        
        try:
            result = exp['function'](quick=args.quick)
            results[exp_name] = {'status': 'success', 'result': result}
        except Exception as e:
            print(f"\n❌ ERROR in {exp_name}: {e}")
            results[exp_name] = {'status': 'error', 'error': str(e)}
        
        elapsed = time.time() - start
        print(f"\n{exp['name']} completed in {elapsed:.1f}s")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for exp_name, res in results.items():
        status = "Success" if res['status'] == 'success' else "Failure"
        print(f"{status} {EXPERIMENTS[exp_name]['name']}")
        
        if res['status'] == 'success' and 'result' in res:
            r = res['result']
            if 'optimal_frequency' in r:
                print(f"   → Optimal frequency: {r['optimal_frequency']:.2f} Hz")
            if 'optimal_range' in r:
                print(f"   → Optimal range: {r['optimal_range']:.0f} px")
            if 'max_safe_agents' in r:
                print(f"   → Max safe agents: {r['max_safe_agents']}")
            if 'cost_reduction' in r:
                print(f"   → Cost reduction with comm: {r['cost_reduction']:.1f}")
    
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Results saved to: results/")
    print()


if __name__ == "__main__":
    main()