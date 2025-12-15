"""
Multi-Agent Path Planning Experiments with Communication

This script runs parameter sweep experiments to evaluate how communication
affects multi-agent path planning performance. All configuration parameters
are centralized in src/config.py.

Available Experiments:
    frequency  - Sweep broadcast frequency (steps per communication)
    msg_length - Sweep message length (waypoints shared)
    2d         - 2D cost landscape (frequency × message length)
    comparison - Compare communication ON vs OFF

Usage:
    python experiments/run_experiments.py              # Run all experiments
    python experiments/run_experiments.py frequency    # Run specific experiment
    python experiments/run_experiments.py --quick      # Quick mode (fewer trials)
    python experiments/run_experiments.py --list       # List experiments
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
from datetime import datetime
import numpy as np

# Import configuration
from src.config import (
    NUM_SEEDS, SIMULATION_FPS, EXPERIMENT_CONFIG, get_experiment_config_value
)

# Import simulation engine
from src.simulation import run_batch_multiple_seeds

# Import visualization
from src.visualization.plots import (
    plot_frequency_sweep,
    plot_range_sweep, 
    plot_cost_landscape_freq_msg
)


# =============================================================================
# EXPERIMENT IMPLEMENTATIONS
# =============================================================================

def run_frequency_sweep(quick=False):
    """
    Sweep over broadcast frequency (defined as steps per communication).
    
    Fixed parameters: comm_range, msg_length, n_agents (from config)
    Sweep parameter: broadcast_interval_steps (min to max from config)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Broadcast Frequency Sweep")
    print("=" * 70)
    
    # Read parameters from config
    min_steps = EXPERIMENT_CONFIG['frequency_sweep_min_steps']
    max_steps = EXPERIMENT_CONFIG['frequency_sweep_max_steps']
    num_values = get_experiment_config_value('frequency_sweep_num_values', quick)
    comm_range = EXPERIMENT_CONFIG['comm_range']
    msg_length = EXPERIMENT_CONFIG['msg_length']
    n_agents = EXPERIMENT_CONFIG['n_agents']
    num_trials = get_experiment_config_value('num_trials', quick)
    
    # Generate sweep values (step intervals)
    step_intervals = np.linspace(min_steps, max_steps, num_values).astype(int)
    frequencies_hz = SIMULATION_FPS / step_intervals  # Convert to Hz for display
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Comm range: {comm_range:.1f}px")
    print(f"  Message length: {msg_length} waypoints")
    print(f"  Trials per point: {num_trials}")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"\nFrequency range:")
    print(f"  Steps: {step_intervals[0]} to {step_intervals[-1]} steps")
    print(f"  Hz: {frequencies_hz[0]:.2f} to {frequencies_hz[-1]:.2f} Hz")
    print(f"  Total points: {len(step_intervals)}\n")
    
    # Run sweep
    results = []
    for i, steps in enumerate(step_intervals):
        freq_hz = frequencies_hz[i]
        print(f"[{i+1:2d}/{len(step_intervals)}] {steps:3d} steps ({freq_hz:5.2f} Hz) ... ", 
              end="", flush=True)
        
        try:
            result = run_batch_multiple_seeds(
                comm_enabled=True,
                broadcast_interval_steps=steps,
                comm_range=comm_range,
                msg_length=msg_length,
                n_agents=n_agents,
                num_trials=num_trials,
                num_seeds=NUM_SEEDS,
                verbose=True  # Enable verbose logging
            )
            
            print(f"cost={result['cost_mean']:6.1f}±{result['cost_std']:4.1f}, "
                  f"safety={result['min_separation_mean']:5.1f}±{result['min_separation_std']:4.1f}px")
            
            result['interval_steps'] = steps
            result['frequency_hz'] = freq_hz
            results.append(result)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(None)
    
    # Filter out failed runs
    results = [r for r in results if r is not None]
    
    if not results:
        print("\n❌ All trials failed!")
        return None
    
    # Plot results
    valid_steps = [r['interval_steps'] for r in results]
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'num_seeds': NUM_SEEDS,
        'comm_range': comm_range,
        'msg_length': msg_length
    }
    plot_frequency_sweep(valid_steps, results, config=config)
    
    # Find optimal
    costs = [r['cost_mean'] for r in results]
    opt_idx = np.argmin(costs)
    
    print(f"\n✓ Optimal: {valid_steps[opt_idx]} steps ({results[opt_idx]['frequency_hz']:.2f} Hz), "
          f"cost={costs[opt_idx]:.1f}")
    
    return {
        'optimal_interval_steps': valid_steps[opt_idx],
        'optimal_frequency_hz': results[opt_idx]['frequency_hz'],
        'optimal_cost': costs[opt_idx],
        'results': results
    }


def run_msg_length_sweep(quick=False):
    """
    Sweep over message length (number of waypoints shared).
    
    Fixed parameters: comm_range, frequency, n_agents (from config)
    Sweep parameter: msg_length (min to max from config)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Message Length Sweep")
    print("=" * 70)
    
    # Read parameters from config
    msg_min = EXPERIMENT_CONFIG['msg_length_sweep_min']
    msg_max = EXPERIMENT_CONFIG['msg_length_sweep_max']
    num_values = get_experiment_config_value('msg_length_sweep_num_values', quick)
    comm_range = EXPERIMENT_CONFIG['comm_range']
    frequency_hz = EXPERIMENT_CONFIG['default_frequency_hz']
    broadcast_interval = int(SIMULATION_FPS / frequency_hz)
    n_agents = EXPERIMENT_CONFIG['n_agents']
    num_trials = get_experiment_config_value('num_trials', quick)
    
    # Generate sweep values
    msg_lengths = np.linspace(msg_min, msg_max, num_values).astype(int)
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Comm range: {comm_range:.1f}px")
    print(f"  Frequency: {frequency_hz:.2f} Hz ({broadcast_interval} steps)")
    print(f"  Trials per point: {num_trials}")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"\nMessage length range:")
    print(f"  {msg_lengths[0]} to {msg_lengths[-1]} waypoints")
    print(f"  Total points: {len(msg_lengths)}\n")
    
    # Run sweep
    results = []
    for i, msg_len in enumerate(msg_lengths):
        print(f"[{i+1:2d}/{len(msg_lengths)}] msg_len={msg_len:2d} ... ", 
              end="", flush=True)
        
        try:
            result = run_batch_multiple_seeds(
                comm_enabled=True,
                broadcast_interval_steps=broadcast_interval,
                comm_range=comm_range,
                msg_length=msg_len,
                n_agents=n_agents,
                num_trials=num_trials,
                num_seeds=NUM_SEEDS,
                verbose=False
            )
            
            print(f"cost={result['cost_mean']:6.1f}±{result['cost_std']:4.1f}, "
                  f"safety={result['min_separation_mean']:5.1f}±{result['min_separation_std']:4.1f}px")
            
            result['msg_length'] = msg_len
            results.append(result)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(None)
    
    # Filter out failed runs
    results = [r for r in results if r is not None]
    
    if not results:
        print("\n❌ All trials failed!")
        return None
    
    # Plot results (reuse range_sweep plot)
    valid_msg_lengths = [r['msg_length'] for r in results]
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'num_seeds': NUM_SEEDS,
        'frequency': frequency_hz,
        'comm_range': comm_range
    }
    plot_range_sweep(valid_msg_lengths, results, 
                     save_path="results/msg_length_sweep.png", config=config)
    
    # Find optimal
    costs = [r['cost_mean'] for r in results]
    opt_idx = np.argmin(costs)
    
    print(f"\n✓ Optimal: {valid_msg_lengths[opt_idx]} waypoints, "
          f"cost={costs[opt_idx]:.1f}")
    
    return {
        'optimal_msg_length': valid_msg_lengths[opt_idx],
        'optimal_cost': costs[opt_idx],
        'results': results
    }


def run_2d_landscape(quick=False):
    """
    2D cost landscape over frequency × message length.
    
    Fixed parameters: comm_range, n_agents (from config)
    Sweep parameters: broadcast_interval_steps, msg_length (from config)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: 2D Cost Landscape (Frequency × Message Length)")
    print("=" * 70)
    
    # Read parameters from config
    freq_min = EXPERIMENT_CONFIG['landscape_freq_min_steps']
    freq_max = EXPERIMENT_CONFIG['landscape_freq_max_steps']
    freq_num = get_experiment_config_value('landscape_freq_num_values', quick)
    msg_min = EXPERIMENT_CONFIG['landscape_msg_length_min']
    msg_max = EXPERIMENT_CONFIG['landscape_msg_length_max']
    msg_num = get_experiment_config_value('landscape_msg_length_num_values', quick)
    comm_range = EXPERIMENT_CONFIG['comm_range']
    n_agents = EXPERIMENT_CONFIG['n_agents']
    num_trials = get_experiment_config_value('landscape_num_trials', quick)
    
    # Generate sweep values
    step_intervals = np.linspace(freq_min, freq_max, freq_num).astype(int)
    frequencies_hz = SIMULATION_FPS / step_intervals
    msg_lengths = np.linspace(msg_min, msg_max, msg_num).astype(int)
    
    total = len(step_intervals) * len(msg_lengths)
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Comm range: {comm_range:.1f}px")
    print(f"  Trials per point: {num_trials}")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"\nGrid dimensions:")
    print(f"  Frequency: {len(step_intervals)} points ({step_intervals[0]}-{step_intervals[-1]} steps)")
    print(f"  Message length: {len(msg_lengths)} points ({msg_lengths[0]}-{msg_lengths[-1]} waypoints)")
    print(f"  Total combinations: {total}\n")
    
    # Run sweep
    cost_matrix = np.zeros((len(step_intervals), len(msg_lengths)))
    count = 0
    
    for i, steps in enumerate(step_intervals):
        freq_hz = frequencies_hz[i]
        for j, msg_len in enumerate(msg_lengths):
            count += 1
            print(f"[{count:3d}/{total}] {steps:3d} steps ({freq_hz:5.2f} Hz), "
                  f"msg_len={msg_len:2d} ... ", end="", flush=True)
            
            try:
                result = run_batch_multiple_seeds(
                    comm_enabled=True,
                    broadcast_interval_steps=steps,
                    comm_range=comm_range,
                    msg_length=msg_len,
                    n_agents=n_agents,
                    num_trials=num_trials,
                    num_seeds=NUM_SEEDS,
                    verbose=False
                )
                
                cost_matrix[i, j] = result['cost_mean']
                print(f"cost={result['cost_mean']:6.1f}±{result['cost_std']:4.1f}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                cost_matrix[i, j] = np.nan
    
    # Check if we have enough valid data
    valid_ratio = np.sum(~np.isnan(cost_matrix)) / cost_matrix.size
    if valid_ratio < 0.5:
        print(f"\n⚠️  Warning: Only {valid_ratio:.0%} of trials succeeded")
    
    # Plot results
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'num_seeds': NUM_SEEDS,
        'comm_range': comm_range
    }
    plot_cost_landscape_freq_msg(frequencies_hz, msg_lengths, cost_matrix, config=config)
    
    # Find optimal (ignoring NaN)
    valid_mask = ~np.isnan(cost_matrix)
    if valid_mask.any():
        min_idx = np.unravel_index(np.nanargmin(cost_matrix), cost_matrix.shape)
        opt_steps = step_intervals[min_idx[0]]
        opt_freq = frequencies_hz[min_idx[0]]
        opt_msg = msg_lengths[min_idx[1]]
        opt_cost = cost_matrix[min_idx]
        
        print(f"\n✓ Optimal: {opt_steps} steps ({opt_freq:.2f} Hz), "
              f"{opt_msg} waypoints, cost={opt_cost:.1f}")
        
        return {
            'optimal_interval_steps': opt_steps,
            'optimal_frequency_hz': opt_freq,
            'optimal_msg_length': opt_msg,
            'optimal_cost': opt_cost,
            'cost_matrix': cost_matrix
        }
    else:
        print("\n❌ No valid results!")
        return None


def run_comparison(quick=False):
    """
    Compare communication ON vs OFF.
    
    Uses default parameters from config for both conditions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Communication ON vs OFF Comparison")
    print("=" * 70)
    
    # Read parameters from config
    n_agents = EXPERIMENT_CONFIG['n_agents']
    num_trials = get_experiment_config_value('comparison_num_trials', quick)
    frequency_hz = EXPERIMENT_CONFIG['default_frequency_hz']
    broadcast_interval = int(SIMULATION_FPS / frequency_hz)
    comm_range = EXPERIMENT_CONFIG['comm_range']
    msg_length = EXPERIMENT_CONFIG['msg_length']
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Frequency: {frequency_hz:.2f} Hz ({broadcast_interval} steps)")
    print(f"  Comm range: {comm_range:.1f}px")
    print(f"  Message length: {msg_length} waypoints")
    print(f"  Trials: {num_trials}")
    print(f"  Seeds: {NUM_SEEDS}\n")
    
    # Run WITH communication
    print("Running WITH communication ... ", end="", flush=True)
    try:
        result_on = run_batch_multiple_seeds(
            comm_enabled=True,
            broadcast_interval_steps=broadcast_interval,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            num_seeds=NUM_SEEDS,
            verbose=False
        )
        print(f"cost={result_on['cost_mean']:.1f}±{result_on['cost_std']:.1f}, "
              f"collisions={result_on['collision_rate']:.1%}±{result_on['collision_rate_std']:.1%}")
    except Exception as e:
        print(f"FAILED: {e}")
        result_on = None
    
    # Run WITHOUT communication
    print("Running WITHOUT communication ... ", end="", flush=True)
    try:
        result_off = run_batch_multiple_seeds(
            comm_enabled=False,
            broadcast_interval_steps=broadcast_interval,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            num_seeds=NUM_SEEDS,
            verbose=False
        )
        print(f"cost={result_off['cost_mean']:.1f}±{result_off['cost_std']:.1f}, "
              f"collisions={result_off['collision_rate']:.1%}±{result_off['collision_rate_std']:.1%}")
    except Exception as e:
        print(f"FAILED: {e}")
        result_off = None
    
    if result_on is None or result_off is None:
        print("\n❌ Comparison failed!")
        return None
    
    # Print comparison
    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'Comm ON':<15} {'Comm OFF':<15} {'Δ'}")
    print("-" * 70)
    
    cost_delta = result_off['cost_mean'] - result_on['cost_mean']
    print(f"{'Cost':<25} {result_on['cost_mean']:<15.1f} {result_off['cost_mean']:<15.1f} "
          f"{cost_delta:+.1f}")
    
    print(f"{'Collision Rate':<25} {result_on['collision_rate']:<15.1%} "
          f"{result_off['collision_rate']:<15.1%}")
    
    print(f"{'Min Separation (px)':<25} {result_on['min_separation_mean']:<15.1f} "
          f"{result_off['min_separation_mean']:<15.1f}")
    
    print(f"{'Replans':<25} {result_on['replan_mean']:<15.1f} "
          f"{result_off['replan_mean']:<15.1f}")
    
    print("-" * 70)
    
    improvement = (cost_delta / result_off['cost_mean']) * 100 if result_off['cost_mean'] > 0 else 0
    print(f"Communication reduces cost by {improvement:.1f}%")
    
    return {
        'comm_on': result_on,
        'comm_off': result_off,
        'cost_reduction': cost_delta,
        'cost_reduction_pct': improvement
    }


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================

EXPERIMENTS = {
    'frequency': {
        'name': 'Frequency Sweep',
        'description': 'Sweep broadcast frequency (steps per communication)',
        'function': run_frequency_sweep
    },
    'msg_length': {
        'name': 'Message Length Sweep',
        'description': 'Sweep message length (waypoints shared)',
        'function': run_msg_length_sweep
    },
    '2d': {
        'name': '2D Cost Landscape',
        'description': '2D sweep over frequency × message length',
        'function': run_2d_landscape
    },
    'comparison': {
        'name': 'Communication Comparison',
        'description': 'Compare communication ON vs OFF',
        'function': run_comparison
    }
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Path Planning Experiments with Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all experiments
  %(prog)s frequency          # Run frequency sweep only
  %(prog)s frequency 2d       # Run multiple experiments
  %(prog)s --quick            # Quick mode (fewer trials)
  %(prog)s --list             # List available experiments

All configuration parameters are defined in src/config.py (EXPERIMENT_CONFIG).
        """
    )
    
    parser.add_argument('experiments', nargs='*', 
                       help='Experiments to run (default: all)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available experiments')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode (fewer trials/points)')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 70)
        for key, exp in EXPERIMENTS.items():
            print(f"  {key:<12} {exp['name']}")
            print(f"  {' '*12} {exp['description']}")
        print()
        return
    
    # Determine which experiments to run
    if args.experiments:
        to_run = []
        for name in args.experiments:
            if name in EXPERIMENTS:
                to_run.append(name)
            else:
                print(f"❌ Unknown experiment: {name}")
                print(f"   Available: {', '.join(EXPERIMENTS.keys())}")
                print(f"   Use --list to see descriptions")
                return
    else:
        to_run = list(EXPERIMENTS.keys())
    
    # Header
    print("\n" + "=" * 70)
    print("MULTI-AGENT PATH PLANNING EXPERIMENTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Experiments: {', '.join(to_run)}")
    print("=" * 70)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for exp_name in to_run:
        exp = EXPERIMENTS[exp_name]
        start_time = time.time()
        
        try:
            result = exp['function'](quick=args.quick)
            elapsed = time.time() - start_time
            
            if result is not None:
                results[exp_name] = {'status': 'success', 'result': result, 'time': elapsed}
                print(f"\n✓ {exp['name']} completed in {elapsed:.1f}s")
            else:
                results[exp_name] = {'status': 'failed', 'time': elapsed}
                print(f"\n❌ {exp['name']} failed after {elapsed:.1f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results[exp_name] = {'status': 'error', 'error': str(e), 'time': elapsed}
            print(f"\n❌ {exp['name']} error after {elapsed:.1f}s: {e}")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"Completed: {success_count}/{len(results)} experiments")
    print(f"Total time: {total_elapsed:.1f}s\n")
    
    for exp_name, res in results.items():
        exp = EXPERIMENTS[exp_name]
        status_icon = "✓" if res['status'] == 'success' else "❌"
        print(f"{status_icon} {exp['name']:<25} ({res['time']:.1f}s)")
        
        if res['status'] == 'success' and 'result' in res:
            r = res['result']
            if 'optimal_frequency_hz' in r:
                print(f"   → Optimal: {r['optimal_interval_steps']} steps "
                      f"({r['optimal_frequency_hz']:.2f} Hz)")
            if 'optimal_msg_length' in r:
                print(f"   → Optimal: {r['optimal_msg_length']} waypoints")
            if 'cost_reduction_pct' in r:
                print(f"   → Communication reduces cost by {r['cost_reduction_pct']:.1f}%")
        elif res['status'] == 'error':
            print(f"   → Error: {res['error']}")
    
    print(f"\nResults saved to: results/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
