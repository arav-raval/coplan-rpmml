"""Parameter sweep over communication range."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.simulation import run_batch
from src.visualization.plots import plot_range_sweep


def main():
    print("=" * 60)
    print("COMMUNICATION RANGE SWEEP EXPERIMENT")
    print("=" * 60)
    
    ranges = np.linspace(50, 400, 15)  
    
    frequency = 1.0         
    msg_length = 10         
    n_agents = 4            
    num_trials = 10
    
    results = []
    
    print(f"Sweeping {len(ranges)} range values: {ranges[0]:.0f} to {ranges[-1]:.0f} px")
    print(f"Agents: {n_agents} | Trials per value: {num_trials} | Frequency: {frequency} Hz\n")
    
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
        
        print(f"cost={result['cost_mean']:6.1f}, replans={result['replan_mean']:4.1f}, "
              f"safety={result['min_separation_mean']:5.1f}px, collisions={result['collision_rate']:.0%}")
        
        result['range'] = r
        results.append(result)
    
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'frequency': frequency,
        'msg_length': msg_length,
    }
    
    print("\nGenerating plots...")
    plot_range_sweep(ranges, results, config=config)
    
    costs = [r['cost_mean'] for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    
    print("\n" + "=" * 60)
    print("RESULTS INTERPRETATION")
    print("=" * 60)
    
    zero_collision_ranges = [ranges[i] for i, cr in enumerate(collision_rates) if cr == 0]
    if zero_collision_ranges:
        print(f"\n MINIMUM SAFE RANGE: {min(zero_collision_ranges):.0f}px")
        print(f"   (Smallest range with 0% collision rate)")
    
    min_idx = np.argmin(costs)
    print(f"\n OPTIMAL RANGE: {ranges[min_idx]:.0f}px")
    print(f"   Cost: {costs[min_idx]:.1f}")
    print(f"   Collision rate: {results[min_idx]['collision_rate']:.0%}")


if __name__ == "__main__":
    main()