"""2D parameter sweep: frequency vs message length."""

import sys
sys.path.insert(0, '.')

import argparse
import time
import numpy as np
from src.simulation import run_batch_multiple_seeds
from src.visualization.plots import plot_cost_landscape_freq_msg
from src.config import EXPERIMENT_CONFIG, NUM_SEEDS, NUM_SEEDS_QUICK, SIMULATION_FPS


def main():
    parser = argparse.ArgumentParser(description="2D Cost Landscape: Frequency × Message Length")
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer values')
    args = parser.parse_args()
    
    print("=" * 75)
    print("2D COST LANDSCAPE: FREQUENCY × MESSAGE LENGTH")
    print("=" * 75)
    
    # Get configuration
    n_agents = EXPERIMENT_CONFIG['n_agents']
    comm_range = EXPERIMENT_CONFIG['comm_range']
    
    if args.quick:
        freq_min = EXPERIMENT_CONFIG['landscape_freq_min_steps']
        freq_max = EXPERIMENT_CONFIG['landscape_freq_max_steps']
        freq_num = EXPERIMENT_CONFIG['landscape_freq_num_values_quick']
        msg_min = EXPERIMENT_CONFIG['landscape_msg_length_min']
        msg_max = EXPERIMENT_CONFIG['landscape_msg_length_max']
        msg_num = EXPERIMENT_CONFIG['landscape_msg_length_num_values_quick']
        num_seeds = NUM_SEEDS_QUICK
    else:
        freq_min = EXPERIMENT_CONFIG['landscape_freq_min_steps']
        freq_max = EXPERIMENT_CONFIG['landscape_freq_max_steps']
        freq_num = EXPERIMENT_CONFIG['landscape_freq_num_values']
        msg_min = EXPERIMENT_CONFIG['landscape_msg_length_min']
        msg_max = EXPERIMENT_CONFIG['landscape_msg_length_max']
        msg_num = EXPERIMENT_CONFIG['landscape_msg_length_num_values']
        num_seeds = NUM_SEEDS
    
    # Generate parameter grids
    freq_steps = np.linspace(freq_min, freq_max, freq_num, dtype=int)
    freq_steps = sorted(set(freq_steps))  
    msg_lengths = np.linspace(msg_min, msg_max, msg_num, dtype=int)
    msg_lengths = sorted(set(msg_lengths))  
    
    # Initialize matrices for all metrics
    cost_matrix = np.zeros((len(msg_lengths), len(freq_steps)))
    replan_matrix = np.zeros((len(msg_lengths), len(freq_steps)))
    separation_matrix = np.zeros((len(msg_lengths), len(freq_steps)))
    collision_count_matrix = np.zeros((len(msg_lengths), len(freq_steps)))
    collision_rate_matrix = np.zeros((len(msg_lengths), len(freq_steps)))
    
    total = len(msg_lengths) * len(freq_steps)
    count = 0
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Communication range: {comm_range:.0f}px (fixed)")
    print(f"  Frequency range: {freq_min} to {freq_max} steps ({len(freq_steps)} values)")
    print(f"  Message length range: {msg_min} to {msg_max} waypoints ({len(msg_lengths)} values)")
    print(f"  Total combinations: {total}")
    print()
    
    start_time = time.time()
    
    for i, msg_len in enumerate(msg_lengths):
        for j, freq_step in enumerate(freq_steps):
            count += 1
            freq_hz = SIMULATION_FPS / freq_step
            print(f"[{count:3d}/{total}] freq={freq_step:2d} steps ({freq_hz:.2f}Hz), msg={msg_len:2d} wp ... ", 
                  end="", flush=True)
            
            result = run_batch_multiple_seeds(
                comm_enabled=True,
                broadcast_interval_steps=freq_step,
                comm_range=comm_range,
                msg_length=msg_len,
                n_agents=n_agents,
                num_seeds=num_seeds,
                verbose=False
            )
            
            cost_matrix[i, j] = result['cost_mean']
            replan_matrix[i, j] = result['replan_mean']
            separation_matrix[i, j] = result['avg_separation_mean']
            collision_count_matrix[i, j] = result['collision_count_mean']
            collision_rate_matrix[i, j] = result.get('collision_rate', 0)
            print(f"cost={result['cost_mean']:6.1f}")
    
    elapsed = time.time() - start_time
    
    # Store config for plotting
    config = {
        'n_agents': n_agents,
        'num_seeds': num_seeds,
        'comm_range': comm_range,
    }

    print("\nGenerating cost landscape plot...")
    plot_cost_landscape_freq_msg(
        freq_steps, 
        msg_lengths, 
        cost_matrix, 
        save_path="results/cost_landscape_freq_msg.png",
        config=config
    )
    
    print("\nGenerating detailed 4-panel analysis...")
    from src.visualization.plots import plot_2d_sweep_details
    results_matrix = {
        'cost': cost_matrix,
        'replan': replan_matrix,
        'avg_separation': separation_matrix,
        'collision_count': collision_count_matrix,
        'collision_rate': collision_rate_matrix
    }
    plot_2d_sweep_details(
        freq_steps,
        msg_lengths,
        results_matrix,
        save_path="results/2d_sweep_details.png",
        config=config
    )
    
    # Find optimal
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    opt_msg = msg_lengths[min_idx[0]]
    opt_freq = freq_steps[min_idx[1]]
    opt_freq_hz = SIMULATION_FPS / opt_freq
    
    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)
    print(f"\nOPTIMAL PARAMETERS:")
    print(f"   Frequency: {opt_freq} steps ({opt_freq_hz:.2f} Hz)")
    print(f"   Message Length: {opt_msg} waypoints")
    print(f"   Cost: {cost_matrix[min_idx]:.1f}")
    print(f"\n✓ Completed in {elapsed:.1f}s")
    print(f"✓ Saved cost landscape: results/cost_landscape_freq_msg.png")
    print(f"✓ Saved detailed analysis: results/2d_sweep_details.png")
    
    return {
        'optimal_frequency_steps': opt_freq,
        'optimal_frequency_hz': opt_freq_hz,
        'optimal_msg_length': opt_msg,
        'optimal_cost': float(cost_matrix[min_idx]),
        'cost_matrix': cost_matrix.tolist()
    }


if __name__ == "__main__":
    main()