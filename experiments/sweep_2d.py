"""2D parameter sweep: frequency vs range."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.simulation import run_batch
from src.visualization.plots import plot_cost_landscape


def main():
    print("=" * 60)
    print("2D COST LANDSCAPE: FREQUENCY vs RANGE")
    print("=" * 60)
    print("""
    WHAT WE'RE TESTING:
    - Simultaneously varying frequency AND range
    - Looking for the optimal combination
    - Identifying trade-offs between the two parameters
    
    QUESTION: What (frequency, range) pair minimizes cost?
    """)
    
    # Parameters to sweep
    frequencies = np.linspace(0.5, 3.0, 8)
    ranges = np.linspace(100, 350, 8)
    
    # Fixed parameters
    msg_length = 10
    n_agents = 4
    num_trials = 5  # Fewer trials for speed (64 combos × 5 = 320 runs)
    
    cost_matrix = np.zeros((len(frequencies), len(ranges)))
    
    total = len(frequencies) * len(ranges)
    count = 0
    
    print(f"\nRunning {total} parameter combinations...")
    print(f"Frequencies: {frequencies[0]:.2f} to {frequencies[-1]:.2f} Hz ({len(frequencies)} values)")
    print(f"Ranges: {ranges[0]:.0f} to {ranges[-1]:.0f} px ({len(ranges)} values)")
    print(f"Agents: {n_agents} | Trials per combo: {num_trials}\n")
    
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
    
    # Store config for plotting
    config = {
        'n_agents': n_agents,
        'num_trials': num_trials,
        'msg_length': msg_length,
    }

    print("\nGenerating cost landscape plot...")
    plot_cost_landscape(frequencies, ranges, cost_matrix, config=config)
    
    # Find optimal
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOPTIMAL PARAMETERS:")
    print(f"   Frequency: {frequencies[min_idx[0]]:.2f} Hz")
    print(f"   Range: {ranges[min_idx[1]]:.0f} px")
    print(f"   Cost: {cost_matrix[min_idx]:.1f}")
    
    print(f"\nCOST MATRIX (rows=freq, cols=range):")
    print(f"       ", end="")
    for r in ranges:
        print(f"{r:6.0f}", end="")
    print()
    for i, freq in enumerate(frequencies):
        print(f"{freq:4.2f}Hz ", end="")
        for j in range(len(ranges)):
            print(f"{cost_matrix[i,j]:6.1f}", end="")
        print()


if __name__ == "__main__":
    main()