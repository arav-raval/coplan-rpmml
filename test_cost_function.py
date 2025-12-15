"""
Test script to demonstrate the refined cost function.
Shows cost breakdown for different parameter combinations.
"""

import numpy as np
from src.metrics.collector import SimulationMetrics
from src.config import WEIGHTS, SAFETY_THRESHOLD, SIMULATION_FPS

def create_test_scenario(freq_steps, msg_length, min_sep, time_steps, 
                         messages=10, replans=5, collisions=0, completed=True):
    """Create a test scenario with specific parameters."""
    freq_hz = SIMULATION_FPS / freq_steps
    
    metrics = SimulationMetrics()
    metrics.total_time = time_steps
    metrics.agent1_distance = 500.0
    metrics.agent2_distance = 500.0
    metrics.min_separation = min_sep
    metrics.avg_separation = min_sep + 20  # Typically higher than min
    metrics.separation_samples = time_steps
    metrics.collision_count = collisions
    metrics.collision_occurred = collisions > 0
    metrics.replan_count = replans
    metrics.messages_sent = messages
    metrics.both_reached_goal = completed
    metrics.timed_out = not completed
    
    # Communication parameters
    metrics.broadcast_frequency = freq_hz
    metrics.comm_range = 250.0
    metrics.msg_length = msg_length
    
    return metrics

def print_cost_breakdown(name, metrics):
    """Print detailed cost breakdown for a scenario."""
    breakdown = metrics.compute_cost_breakdown()
    total = breakdown['total']
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  Frequency: {metrics.broadcast_frequency:.2f} Hz "
          f"({SIMULATION_FPS/metrics.broadcast_frequency:.0f} steps)")
    print(f"  Message Length: {metrics.msg_length} waypoints")
    print(f"  Messages Sent: {metrics.messages_sent}")
    print(f"  Replans: {metrics.replan_count}")
    print(f"  Min Separation: {metrics.min_separation:.1f} px")
    print(f"  Collisions: {metrics.collision_count}")
    print(f"  Completed: {'Yes' if metrics.both_reached_goal else 'No'}")
    print(f"  Time: {metrics.total_time:.0f} steps")
    
    print(f"\nCost Breakdown:")
    for component, value in breakdown.items():
        if component != 'total':
            pct = (value / total * 100) if total > 0 else 0
            print(f"  {component:12s}: {value:8.2f}  ({pct:5.1f}%)")
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':12s}: {total:8.2f}")
    
    return total

def main():
    print("\n" + "="*70)
    print("REFINED COST FUNCTION TEST")
    print("="*70)
    print(f"\nWeights:")
    for k, v in WEIGHTS.items():
        print(f"  {k:12s}: {v}")
    print(f"\nSafety Threshold: {SAFETY_THRESHOLD} px")
    
    scenarios = []
    
    # Scenario 1: High frequency, short messages (communicate every step)
    print("\n" + "="*70)
    print("SCENARIO 1: High Frequency + Short Messages")
    print("="*70)
    m1 = create_test_scenario(
        freq_steps=1,      # 60 Hz - every step
        msg_length=4,      # Short messages
        min_sep=15.0,      # Safe separation
        time_steps=120,
        messages=100,      # Many messages (high freq)
        replans=10,
        collisions=0,
        completed=True
    )
    c1 = print_cost_breakdown("High Freq + Short Msg (Safe)", m1)
    scenarios.append(("High Freq + Short", c1))
    
    # Scenario 2: Medium frequency, medium messages (communicate every 5 steps)
    print("\n" + "="*70)
    print("SCENARIO 2: Medium Frequency + Medium Messages")
    print("="*70)
    m2 = create_test_scenario(
        freq_steps=5,      # 12 Hz
        msg_length=10,     # Medium messages
        min_sep=12.0,      # Still safe
        time_steps=120,
        messages=20,       # Fewer messages
        replans=8,
        collisions=0,
        completed=True
    )
    c2 = print_cost_breakdown("Medium Freq + Medium Msg (Safe)", m2)
    scenarios.append(("Medium Freq + Medium", c2))
    
    # Scenario 3: Low frequency, long messages (communicate every 30 steps)
    print("\n" + "="*70)
    print("SCENARIO 3: Low Frequency + Long Messages")
    print("="*70)
    m3 = create_test_scenario(
        freq_steps=30,     # 2 Hz
        msg_length=20,     # Long messages
        min_sep=8.0,       # Getting close (< threshold)
        time_steps=125,    # Slightly slower
        messages=5,        # Very few messages
        replans=4,
        collisions=0,
        completed=True
    )
    c3 = print_cost_breakdown("Low Freq + Long Msg (Risky)", m3)
    scenarios.append(("Low Freq + Long", c3))
    
    # Scenario 4: Very low frequency (collision)
    print("\n" + "="*70)
    print("SCENARIO 4: Very Low Frequency (Collision)")
    print("="*70)
    m4 = create_test_scenario(
        freq_steps=60,     # 1 Hz
        msg_length=25,     # Very long messages
        min_sep=-5.0,      # Collision territory
        time_steps=130,
        messages=3,
        replans=2,
        collisions=2,      # Multiple collisions!
        completed=True
    )
    c4 = print_cost_breakdown("Very Low Freq (Collision!)", m4)
    scenarios.append(("Very Low Freq (Collision)", c4))
    
    # Scenario 5: High frequency but failed to complete
    print("\n" + "="*70)
    print("SCENARIO 5: High Frequency but Timeout")
    print("="*70)
    m5 = create_test_scenario(
        freq_steps=1,      # 60 Hz
        msg_length=4,
        min_sep=20.0,      # Safe
        time_steps=300,    # Took too long
        messages=250,      # Lots of messages
        replans=30,        # Many replans
        collisions=0,
        completed=False    # Didn't finish!
    )
    c5 = print_cost_breakdown("High Freq but Timeout", m5)
    scenarios.append(("High Freq (Timeout)", c5))
    
    # Summary comparison
    print("\n" + "="*70)
    print("COST COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Scenario':<30s} {'Total Cost':>12s}")
    print("─" * 70)
    for name, cost in sorted(scenarios, key=lambda x: x[1]):
        print(f"{name:<30s} {cost:>12.2f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Communication cost now scales with actual usage (messages × length)")
    print("2. Risk penalty kicks in when min_separation < 10px threshold")
    print("3. Collisions have very high cost (3000 per event)")
    print("4. Timeout penalty applied if agents don't complete")
    print("5. Optimal strategy balances communication cost vs safety risk")
    print("\nTo tune: Run experiments and adjust weights so each")
    print("         active term contributes comparably to total cost.")
    print("="*70)

if __name__ == "__main__":
    main()
