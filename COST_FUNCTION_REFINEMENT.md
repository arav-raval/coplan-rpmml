# Cost Function Refinement

## Summary

The cost function has been refined to create **meaningful trade-offs** between communication overhead and safety. The previous version always preferred "communicate every step" because communication costs were too small relative to safety gains.

## New Cost Structure

```python
cost = w_time * total_time
     + w_freq * broadcast_frequency_hz        # Base readiness overhead (small)
     + w_msg * (messages_sent * msg_length)   # Actual data transmitted
     + w_replan * replan_count                # Per-replan overhead
     + w_risk * (gap/threshold)^2             # Safety shaping (pre-collision)
     + w_collision * collision_count          # Large penalty per collision
     + w_timeout * (1 if timed_out else 0)    # Penalty for non-completion
```

### Key Changes

1. **Distance dropped**: Redundant with time (both measure efficiency)
2. **Actual usage tracking**: `messages_sent * msg_length` charges for real data transmitted
3. **Safety shaping**: Quadratic penalty when `min_separation < 10px` threshold
4. **Collision count**: Uses actual number of collision events (not just occurrence flag)
5. **Completion tracking**: Penalizes scenarios where agents don't reach goals

## Weight Configuration

From `src/config.py`:

```python
# Safety threshold defined relative to agent geometry
AGENT_RADIUS = 15  # pixels
COLLISION_DISTANCE = AGENT_RADIUS * 2.0  # 30px (center-to-center at touching)
SAFETY_THRESHOLD = AGENT_RADIUS * 0.67  # ~10px (edge-to-edge clearance)

WEIGHTS = {
    'time': 1.0,              # Penalize simulation time
    'distance': 0.0,          # Dropped (redundant with time)
    'frequency': 0.75,        # Base overhead for high Hz
    'message': 1.0,           # Cost per waypoint sent
    'replan': 0.75,           # Optional per-replan overhead
    'risk': 200.0,            # Safety shaping for close calls
    'collision': 3000.0,      # Large penalty per collision
    'timeout': 1000.0,        # Penalty if didn't complete
}
```

## Test Results

Running `python test_cost_function.py` shows the cost function creates proper incentives:

| Scenario | Total Cost | Key Metrics |
|----------|------------|-------------|
| **Low Freq + Long Msg** | **237.50** | ✅ Optimal: Minimal comm, stays safe |
| Medium Freq + Medium | 335.00 | Good balance |
| High Freq + Short | 572.50 | Too much communication overhead |
| High Freq (Timeout) | 2367.50 | ❌ Didn't complete |
| Very Low Freq (Collision) | 6657.25 | ❌ Collision penalty dominates |

### Cost Breakdown Example (Low Freq + Long - Optimal)

- Time: 125.00 (52.6%) - Simulation steps
- Message: 100.00 (42.1%) - Actual data sent (5 messages × 20 waypoints)
- Risk: 8.00 (3.4%) - Small penalty for min_sep=8px < 10px threshold
- Frequency: 1.50 (0.6%) - Base overhead for 2 Hz
- Replan: 3.00 (1.3%) - 4 replans
- **Total: 237.50**

### Cost Breakdown Example (High Freq + Short)

- Message: 400.00 (69.9%) - Heavy penalty for 100 messages × 4 waypoints
- Time: 120.00 (21.0%)
- Frequency: 45.00 (7.9%) - Higher base overhead for 60 Hz
- **Total: 572.50** (2.4× higher than optimal!)

## Why This Works

1. **Communication scales with usage**: High frequency with few replans → low cost. High frequency with many replans/long messages → high cost.

2. **Safety shaping**: The `(gap/threshold)^2` term creates a "soft barrier" - agents are incentivized to maintain `min_sep > 10px` without needing actual collisions.

3. **Time vs distance**: No longer double-counting progress.

4. **Collision/timeout penalties**: Safety and completion remain paramount (3000 and 1000 respectively).

## Tuning Procedure

1. **Run test scenarios** with different parameter combinations:
   ```bash
   python test_cost_function.py
   ```

2. **Check experiment results**:
   ```bash
   python experiments/run_experiments.py frequency --quick
   ```

3. **Adjust weights** if needed:
   - If optimum still at `steps=1`: increase `w_message` or `w_frequency`
   - If collisions spike when comms drop: increase `w_risk` or `w_collision`
   - Aim for each active term to contribute comparably (not dominated by one component)

## Impact on Experiments

- **Frequency sweep**: Should now show a U-shaped cost curve
  - Too high frequency → excessive communication cost
  - Too low frequency → risk/collision penalties
  - Sweet spot in the middle

- **Message length sweep**: Should favor shorter messages when frequent communication is needed

- **2D sweep**: Should reveal optimal (frequency, message_length) combinations that balance all factors

## Implementation Details

### Metrics Tracked

- `messages_sent`: Incremented on each actual message transmission
- `both_reached_goal`: True if both agents completed their paths
- `timed_out`: True if simulation exceeded time limit
- `collision_count`: Number of discrete collision events

### Usage in Code

```python
# Compute cost with parameters
cost = metrics.compute_cost()

# Get detailed breakdown
breakdown = metrics.compute_cost_breakdown()
# Returns dict: {'time': ..., 'message': ..., 'risk': ..., 'total': ...}
```

## Next Steps

1. Run full experiment sweeps with new cost function
2. Verify that optimal points make intuitive sense
3. Fine-tune weights based on experimental results
4. Consider adding cost breakdown visualization to plots

---

*Last updated: Based on refined cost structure implementation*
