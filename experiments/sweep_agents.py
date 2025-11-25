"""Parameter sweep over number of agents."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.simulation import run_batch
from src.visualization.plots import plot_agent_sweep


def main():
    print("=" * 60)
    print("AGENT COUNT SWEEP EXPERIMENT")
    print("=" * 60)
    
    # Agent counts to test
    agent_counts = [2, 3, 4, 5, 6, 7, 8]
    
    # Fixed parameters
    frequency = 1.5       # Using a "good" frequency from previous sweep
    comm_range = 250.0
    msg_length = 10
    num_trials = 10
    
    results = []
    
    print(f"Sweeping {len(agent_counts)} agent counts: {agent_counts}")
    print(f"Frequency: {frequency} Hz | Range: {comm_range}px | Trials: {num_trials}\n")
    
    for i, n_agents in enumerate(agent_counts):
        print(f"[{i+1}/{len(agent_counts)}] Agents={n_agents} ...", end=" ", flush=True)
        
        result = run_batch(
            comm_enabled=True,
            broadcast_frequency=frequency,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            num_trials=num_trials,
            verbose=False
        )
        
        print(f"cost={result['cost_mean']:6.1f}, replans={result['replan_mean']:4.1f}, "
              f"safety={result['min_separation_mean']:5.1f}px, collisions={result['collision_rate']:.0%}")
        
        result['n_agents'] = n_agents
        results.append(result)
    
    # Config for plot
    config = {
        'num_trials': num_trials,
        'frequency': frequency,
        'comm_range': comm_range,
        'msg_length': msg_length,
    }
    
    print("\nGenerating plots...")
    plot_agent_sweep(agent_counts, results, config=config)
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    costs = [r['cost_mean'] for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    replans = [r['replan_mean'] for r in results]
    
    print(f"\n{'Agents':<8} {'Cost':<10} {'Replans':<10} {'Collisions':<12} {'Status'}")
    print("-" * 50)
    for i, n in enumerate(agent_counts):
        status = "Safe" if collision_rates[i] == 0 else "⚠️ Risky" if collision_rates[i] < 0.3 else "❌ Unsafe"
        print(f"{n:<8} {costs[i]:<10.1f} {replans[i]:<10.1f} {collision_rates[i]:<12.0%} {status}")
    
    # Find max safe agents
    safe_counts = [agent_counts[i] for i, cr in enumerate(collision_rates) if cr < 0.1]
    if safe_counts:
        print(f"\nMaximum safe agent count (with current settings): {max(safe_counts)}")
    else:
        print(f"\nNo completely safe configurations found!")
    
    # Scaling analysis
    if len(agent_counts) >= 3:
        cost_increase = (costs[-1] - costs[0]) / (agent_counts[-1] - agent_counts[0])
        print(f"\nCost scaling: ~{cost_increase:.1f} cost per additional agent")


if __name__ == "__main__":
    main()