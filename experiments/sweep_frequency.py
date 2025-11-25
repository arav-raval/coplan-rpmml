"""Parameter sweep over broadcast frequency."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.simulation import run_batch
from src.visualization.plots import plot_frequency_sweep, identify_plateaus


def main():
    print("=" * 60)
    print("FREQUENCY SWEEP EXPERIMENT")
    print("=" * 60)
    
    # Parameters to sweep
    frequencies = np.linspace(0.2, 5.0, 15)  # 0.2 Hz to 5 Hz
    
    # Fixed parameters
    comm_range = 250.0      # Fixed: agents can hear each other within 250px
    msg_length = 10         # Fixed: share 10 waypoints per message
    n_agents = 4            # Fixed: 4 agents in simulation
    num_trials = 10         # Run 10 trials per frequency value
    
    results = []
    
    print(f"Sweeping {len(frequencies)} frequency values: {frequencies[0]:.2f} to {frequencies[-1]:.2f} Hz")
    print(f"Agents: {n_agents} | Trials per value: {num_trials} | Comm range: {comm_range}px\n")
    
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
        
        print(f"cost={result['cost_mean']:6.1f}, replans={result['replan_mean']:4.1f}, "
              f"safety={result['min_separation_mean']:5.1f}px, collisions={result['collision_rate']:.0%}")
        
        result['frequency'] = freq
        results.append(result)
    # In sweep_frequency.py, update the plot call:

    # Store config for plotting
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'comm_range': comm_range,
        'msg_length': msg_length,
    }
    
    # Generate plot with config
    print("\nGenerating plots...")
    plot_frequency_sweep(frequencies, results, config=config)
    
    # Identify plateaus
    costs = [r['cost_mean'] for r in results]
    plateaus = identify_plateaus(frequencies, costs)
    
    print("\n" + "=" * 60)
    print("RESULTS INTERPRETATION")
    print("=" * 60)
    
    if plateaus:
        print("\nPLATEAU REGIONS DETECTED:")
        print("   (Regions where increasing frequency doesn't help)")
        for start, end in plateaus:
            print(f"   Frequency: {frequencies[start]:.2f} - {frequencies[end]:.2f} Hz")
            print(f"   Cost: {costs[start]:.1f} - {costs[end]:.1f}")
    else:
        print("\nNo significant plateau regions detected.")
    
    min_idx = np.argmin(costs)
    print(f"\nOPTIMAL FREQUENCY: {frequencies[min_idx]:.2f} Hz")
    print(f"   Cost: {costs[min_idx]:.1f}")
    print(f"   Min separation: {results[min_idx]['min_separation_mean']:.1f}px")
    print(f"   Replan count: {results[min_idx]['replan_mean']:.1f}")
    print(f"   Collision rate: {results[min_idx]['collision_rate']:.0%}")


if __name__ == "__main__":
    main()