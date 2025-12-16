# src/simulation.py
"""Headless simulation runner for experiments."""

import random
import pymunk
import numpy as np
from itertools import combinations

from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MAX_SPEED,
    PREDICTION_HORIZON, PREDICTION_DT, COLLISION_DISTANCE,
    AGENT_RADIUS, DEFAULT_NUM_AGENTS, AGENT_COLORS,
    POSITION_MARGIN, MIN_POSITION_SEPARATION, MIN_TRAVEL_DISTANCE,
    MAX_MSG_LENGTH_STEPS, SIMULATION_FPS, REPLAN_COOLDOWN_STEPS
)
from src.world import create_space, create_boundaries
from src.agents import Agent
from src.metrics import MetricsCollector, SimulationMetrics


def generate_random_positions(n_agents, width, height, margin=None, seed=None):
    """Generate random start and goal positions for agents.
    
    Args:
        n_agents: Number of agents
        width: Screen width
        height: Screen height
        margin: Margin from edges (defaults to POSITION_MARGIN from config)
        seed: Random seed (optional, for deterministic generation)
    """
    if seed is not None:
        random.seed(seed)
    if margin is None:
        margin = POSITION_MARGIN
    min_separation = MIN_POSITION_SEPARATION
    positions = []
    
    for _ in range(n_agents):
        attempts = 0
        while attempts < 100:
            start = (random.randint(margin, width - margin),
                     random.randint(margin, height - margin))
            valid = True
            for existing_start, _ in positions:
                if (abs(start[0] - existing_start[0]) < min_separation and
                    abs(start[1] - existing_start[1]) < min_separation):
                    valid = False
                    break
            if valid:
                break
            attempts += 1
        
        attempts = 0
        while attempts < 100:
            if start[0] < width // 2:
                goal_x = random.randint(width // 2, width - margin)
            else:
                goal_x = random.randint(margin, width // 2)
            if start[1] < height // 2:
                goal_y = random.randint(height // 2, height - margin)
            else:
                goal_y = random.randint(margin, height // 2)
            goal = (goal_x, goal_y)
            
            dist = ((goal[0] - start[0])**2 + (goal[1] - start[1])**2)**0.5
            if dist > MIN_TRAVEL_DISTANCE:
                break
            attempts += 1
        
        positions.append((start, goal))
    
    return positions


def predict_collision_pair(agent1, agent2, horizon_seconds):
    """Check if two agents will collide within the prediction horizon."""
    if not agent1.path or agent1.path_index >= len(agent1.path):
        return False
    if not agent2.path or agent2.path_index >= len(agent2.path):
        return False

    sim_pos1 = pymunk.Vec2d(*agent1.body.position)
    sim_idx1 = agent1.path_index
    sim_pos2 = pymunk.Vec2d(*agent2.body.position)
    sim_idx2 = agent2.path_index

    # Check initial distance (agents might already be very close)
    if sim_pos1.get_distance(sim_pos2) < COLLISION_DISTANCE:
        return True

    t = 0.0
    while t < horizon_seconds:
        if sim_idx1 < len(agent1.path):
            target1 = agent1.path[sim_idx1]
            vec1 = target1 - sim_pos1
            move = MAX_SPEED * PREDICTION_DT
            if vec1.length > 0:
                if move >= vec1.length:
                    sim_pos1 = target1
                    sim_idx1 += 1
                else:
                    sim_pos1 += vec1.normalized() * move

        if sim_idx2 < len(agent2.path):
            target2 = agent2.path[sim_idx2]
            vec2 = target2 - sim_pos2
            move = MAX_SPEED * PREDICTION_DT
            if vec2.length > 0:
                if move >= vec2.length:
                    sim_pos2 = target2
                    sim_idx2 += 1
                else:
                    sim_pos2 += vec2.normalized() * move

        if sim_pos1.get_distance(sim_pos2) < COLLISION_DISTANCE:
            return True

        if sim_idx1 >= len(agent1.path) and sim_idx2 >= len(agent2.path):
            return False

        t += PREDICTION_DT

    return False


def run_simulation(
    comm_enabled: bool = True,
    broadcast_interval_steps: int = 90,
    comm_range: float = 250.0,
    msg_length: int = 10,
    n_agents: int = DEFAULT_NUM_AGENTS,
    seed: int = 5,
    max_time: float = 30.0,
    verbose: bool = False,
    # NEW: Per-agent parameters for MARL (optional, for backward compatibility)
    agent_broadcast_intervals: list = None,
    agent_msg_lengths: list = None,
) -> SimulationMetrics:
    """
    Run a single headless simulation with N agents.
    
    Args:
        comm_enabled: Whether agents communicate
        broadcast_interval_steps: Steps between broadcasts (global, used if agent_broadcast_intervals is None)
        comm_range: Distance at which agents can communicate
        msg_length: Number of waypoints per message (global, used if agent_msg_lengths is None)
        n_agents: Number of agents in simulation
        seed: Random seed for reproducibility
        max_time: Maximum simulation time before timeout
        verbose: Print status messages
        agent_broadcast_intervals: Per-agent broadcast intervals (list of n_agents ints, optional)
        agent_msg_lengths: Per-agent message lengths (list of n_agents ints, optional)
        
    Returns:
        SimulationMetrics with all collected data
    """
    # Handle per-agent parameters (backward compatibility)
    if agent_broadcast_intervals is None:
        agent_broadcast_intervals = [broadcast_interval_steps] * n_agents
    if agent_msg_lengths is None:
        agent_msg_lengths = [msg_length] * n_agents
    
    # Validate per-agent parameter lengths
    if len(agent_broadcast_intervals) != n_agents:
        raise ValueError(f"agent_broadcast_intervals must have length {n_agents}, got {len(agent_broadcast_intervals)}")
    if len(agent_msg_lengths) != n_agents:
        raise ValueError(f"agent_msg_lengths must have length {n_agents}, got {len(agent_msg_lengths)}")
    
    # Warn if message length exceeds prediction horizon (information beyond horizon is irrelevant)
    for i, msg_len in enumerate(agent_msg_lengths):
        if msg_len > MAX_MSG_LENGTH_STEPS:
            if verbose:
                print(f"  [WARNING] Agent {i} msg_length={msg_len} exceeds MAX_MSG_LENGTH_STEPS={MAX_MSG_LENGTH_STEPS}")
                print(f"           Information beyond prediction horizon is irrelevant. Capping to {MAX_MSG_LENGTH_STEPS}.")
    
    # Use configured cooldown value (not tied to communication frequency)
    replan_cooldown_value = REPLAN_COOLDOWN_STEPS
    
    # Setup world
    space = create_space()
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Generate random positions and create agents (seed ensures deterministic generation)
    positions = generate_random_positions(n_agents, SCREEN_WIDTH, SCREEN_HEIGHT, seed=seed)
    
    agents = []
    for i, (start, goal) in enumerate(positions):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent = Agent(space, start, goal, static_obstacles,
                      (SCREEN_WIDTH, SCREEN_HEIGHT), color)
        agent.id = i
        # Store per-agent communication parameters
        agent.broadcast_interval = agent_broadcast_intervals[i]
        agent.msg_length = agent_msg_lengths[i]
        agent.messages_sent_recent = 0  # Track for observations
        agent.replans_recent = 0
        agents.append(agent)
    
    # Initial planning
    if verbose:
        print(f"  Initial planning for {n_agents} agents...")
    
    for i, agent in enumerate(agents):
        agent.plan_path()
        if verbose and agent.path:
            print(f"    Agent {i}: {len(agent.path)} waypoints")
        elif verbose:
            print(f"    Agent {i}: ❌ FAILED")
    
    # Check if any planning failed
    if any(agent.path is None for agent in agents):
        if verbose:
            print(f"  ⚠️  Initial planning failed for some agents")
        metrics = SimulationMetrics()
        metrics.timed_out = True
        return metrics
    
    # Convert broadcast_interval_steps to Hz for MetricsCollector (for cost calculation)
    # Use average across all agents for global metrics
    avg_broadcast_interval = sum(agent_broadcast_intervals) / len(agent_broadcast_intervals)
    avg_msg_length = sum(agent_msg_lengths) / len(agent_msg_lengths)
    broadcast_frequency_hz = SIMULATION_FPS / avg_broadcast_interval if avg_broadcast_interval > 0 else 0.0
    
    # Metrics collection (uses average params for global cost calculation)
    collector = MetricsCollector(
        broadcast_frequency=broadcast_frequency_hz,
        comm_range=comm_range,
        msg_length=int(avg_msg_length)
    )
    
    # Per-agent cooldowns (in steps)
    replan_cooldowns = {i: 0 for i in range(n_agents)}
    
    # Track collision pairs to count events, not frames
    active_collision_pairs = set()  # Pairs currently in collision
    
    # Track last communication step for each agent pair
    last_comm_step = {}  # (i, j) -> step_number
    
    # Debug: Track communication attempts
    comm_checks_total = 0
    comm_checks_allowed = 0
    comm_checks_skipped = 0
    
    # Simulation loop
    dt = 1.0 / SIMULATION_FPS
    current_time = 0.0
    step_count = 0
    
    while current_time < max_time:
        space.step(dt)
        current_time += dt
        step_count += 1
        
        # Update cooldowns (decrement by 1 step)
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= 1
        
        # Communication-based replanning (check ALL pairs)
        if comm_enabled:
            for i, j in combinations(range(n_agents), 2):
                agent_i = agents[i]
                agent_j = agents[j]
                
                # Check which agents are done
                agent_i_done = (agent_i.path is None or agent_i.path_index >= len(agent_i.path))
                agent_j_done = (agent_j.path is None or agent_j.path_index >= len(agent_j.path))
                
                # Skip if BOTH agents are done (no need to check this pair)
                if agent_i_done and agent_j_done:
                    continue
                
                # Check if enough steps have passed since last communication
                pair = (min(i, j), max(i, j))
                
                # Use minimum broadcast interval of the two agents (communication happens at faster rate)
                min_broadcast_interval = min(agent_i.broadcast_interval, agent_j.broadcast_interval)
                last_step = last_comm_step.get(pair, -min_broadcast_interval)  # Allow communication on first check
                
                comm_checks_total += 1
                
                # Bypass timing check if one agent is done (immediate notification needed)
                timing_ok = (step_count - last_step) >= min_broadcast_interval
                immediate_check_needed = agent_i_done or agent_j_done  # Stable agent = immediate check
                
                # Communicate if timing ok OR if immediate check needed
                if timing_ok or immediate_check_needed:
                    comm_checks_allowed += 1
                    pos_i = agent_i.body.position
                    pos_j = agent_j.body.position
                    distance = pos_i.get_distance(pos_j)

                    if distance < comm_range:
                        # Update last communication time for this pair
                        last_comm_step[pair] = step_count
                        
                        # Count this broadcast as a message (actual communication occurred)
                        collector.record_message()
                        # Track per-agent message counts for observations
                        agent_i.messages_sent_recent += 1
                        agent_j.messages_sent_recent += 1
                        
                        if predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON):
                            # Determine which agent should replan and if we should force it (bypass cooldown)
                            force_replan = False
                            
                            # Case 1: One agent is done → active agent MUST replan (forced)
                            if agent_i_done and not agent_j_done:
                                replanner_idx = j
                                replanner = agent_j
                                force_replan = True  # Active agent must avoid stable agent
                            elif agent_j_done and not agent_i_done:
                                replanner_idx = i
                                replanner = agent_i
                                force_replan = True  # Active agent must avoid stable agent
                            
                            # Case 2: Both active - check cooldowns
                            else:
                                agent_i_on_cooldown = replan_cooldowns[i] > 0
                                agent_j_on_cooldown = replan_cooldowns[j] > 0
                                
                                # If one is on cooldown, the other MUST replan (forced)
                                if agent_i_on_cooldown and not agent_j_on_cooldown:
                                    replanner_idx = j
                                    replanner = agent_j
                                    force_replan = True  # Other agent must take responsibility
                                elif agent_j_on_cooldown and not agent_i_on_cooldown:
                                    replanner_idx = i
                                    replanner = agent_i
                                    force_replan = True  # Other agent must take responsibility
                                else:
                                    # Both on cooldown or both available - use convention (higher index)
                                    replanner_idx = j
                                    replanner = agent_j
                                    force_replan = False  # Normal cooldown rules apply
                            
                            # Check cooldown (bypass if forced)
                            if force_replan or replan_cooldowns[replanner_idx] <= 0:
                                if verbose:
                                    force_msg = " [FORCED]" if force_replan else ""
                                    print(f"  [step {step_count:4d}, t={current_time:5.2f}s] Collision predicted: Agent {i} ↔ Agent {j} (d={distance:.1f}px) → Agent {replanner_idx} replans{force_msg}")
                                
                                # Gather paths from agents within communication range only
                                # Include done agents as stationary obstacles (single point at their position)
                                replanner_pos = replanner.body.position
                                # Use replanner's own message length setting
                                replanner_msg_len = replanner.msg_length
                                effective_msg_length = min(replanner_msg_len, MAX_MSG_LENGTH_STEPS) if replanner_msg_len > 0 else MAX_MSG_LENGTH_STEPS
                                obstacles = []
                                for k, other in enumerate(agents):
                                    if k != replanner_idx:
                                        # Check if this agent is within communication range
                                        other_pos = other.body.position
                                        other_distance = replanner_pos.get_distance(other_pos)
                                        if other_distance < comm_range:
                                            # If agent is done, include its final position as obstacle
                                            other_done = (other.path is None or other.path_index >= len(other.path))
                                            if other_done:
                                                # Done agent broadcasts its stationary position
                                                obstacles.append([other_pos])
                                            elif other.path:
                                                # Active agent broadcasts its remaining path
                                                remaining = other.get_remaining_path()
                                                if effective_msg_length > 0 and len(remaining) > effective_msg_length:
                                                    remaining = remaining[:effective_msg_length]
                                                if remaining:
                                                    obstacles.append(remaining)
                                
                                if obstacles:
                                    success = replanner.replan(dynamic_obstacles=obstacles)
                                    if success:
                                        collector.record_replan()
                                        replanner.replans_recent += 1  # Track for observations
                                    # Set cooldown from config (not tied to communication frequency)
                                    replan_cooldowns[replanner_idx] = replan_cooldown_value
                                    # Don't set cooldown on failure - allow immediate retry
                else:
                    comm_checks_skipped += 1
        
        # Update agents
        for agent in agents:
            agent.update()
        
        # Update metrics (track first two agents for compatibility)
        if len(agents) >= 2:
            collector.update(agents[0], agents[1], current_time, AGENT_RADIUS * 2)
            
            # Also track minimum separation across ALL pairs
            # Store CENTER-TO-CENTER distance for collision detection
            # But convert to EDGE-TO-EDGE for safety margin reporting
            for i, j in combinations(range(n_agents), 2):
                center_to_center = agents[i].body.position.get_distance(agents[j].body.position)
                
                # Convert to edge-to-edge clearance (actual gap between agents)
                edge_to_edge = center_to_center - (2 * AGENT_RADIUS)
                
                if edge_to_edge < collector.metrics.min_separation:
                    collector.metrics.min_separation = edge_to_edge
                
                # Track average separation across all pairs and timesteps
                collector.metrics.avg_separation = (
                    (collector.metrics.avg_separation * collector.metrics.separation_samples + edge_to_edge) /
                    (collector.metrics.separation_samples + 1)
                )
                collector.metrics.separation_samples += 1
                
                # Collision check uses center-to-center
                # Count collision EVENTS, not frames
                pair = (min(i, j), max(i, j))
                if center_to_center < AGENT_RADIUS * 2:
                    if not collector.metrics.collision_occurred:
                        # First collision ever
                        collector.metrics.collision_occurred = True
                    # Only increment if this is a NEW collision for this pair
                    if pair not in active_collision_pairs:
                        collector.metrics.collision_count += 1
                        active_collision_pairs.add(pair)
                else:
                    # Agents separated - remove from active collisions
                    active_collision_pairs.discard(pair)
        
        # Check if all agents done
        all_done = all(
            agent.path is None or agent.path_index >= len(agent.path)
            for agent in agents
        )
        if all_done:
            collector.metrics.both_reached_goal = True
            break
    
    timed_out = current_time >= max_time and not collector.metrics.both_reached_goal
    
    # Debug logging
    if verbose and comm_enabled:
        skip_rate = (comm_checks_skipped / comm_checks_total * 100) if comm_checks_total > 0 else 0
        print(f"\n  Communication stats:")
        print(f"    Interval: {broadcast_interval_steps} steps")
        print(f"    Total checks: {comm_checks_total}")
        print(f"    Allowed: {comm_checks_allowed} ({100-skip_rate:.1f}%)")
        print(f"    Skipped: {comm_checks_skipped} ({skip_rate:.1f}%)")
        print(f"    Replans: {collector.metrics.replan_count}")
    
    return collector.finalize(timed_out=timed_out)


def run_batch(
    comm_enabled: bool,
    broadcast_interval_steps: int,
    comm_range: float,
    msg_length: int,
    n_agents: int = DEFAULT_NUM_AGENTS,
    num_trials: int = 10,
    seed_start: int = 0,
    verbose: bool = False
) -> dict:
    """Run multiple trials and aggregate results."""
    
    results = []
    for i in range(num_trials):
        seed = seed_start + i
        if verbose:
            print(f"\n  Trial {i+1}/{num_trials} (seed={seed})")
            print(f"  {'-'*60}")
        
        metrics = run_simulation(
            comm_enabled=comm_enabled,
            broadcast_interval_steps=broadcast_interval_steps,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            seed=seed,
            verbose=verbose
        )
        
        if verbose:
            print(f"  Result: time={metrics.total_time:.2f}s, replans={metrics.replan_count}, "
                  f"collisions={'Yes' if metrics.collision_occurred else 'No'}")
        
        results.append(metrics)
    
    # Aggregate (filter out inf values from failed runs)
    times = [m.total_time for m in results]
    distances = [m.total_distance() for m in results]
    replans = [m.replan_count for m in results]
    separations = [m.min_separation if m.min_separation != float('inf') else 0 for m in results]
    avg_separations = [m.avg_separation if m.separation_samples > 0 else 0 for m in results]
    collisions = [1 if m.collision_occurred else 0 for m in results]
    collision_counts = [m.collision_count for m in results]
    costs = [m.compute_cost() for m in results]

    return {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances),
        'replan_mean': np.mean(replans),
        'replan_std': np.std(replans),
        'min_separation_mean': np.mean(separations),
        'min_separation_std': np.std(separations),
        'avg_separation_mean': np.mean(avg_separations),
        'avg_separation_std': np.std(avg_separations),
        'collision_rate': np.mean(collisions),
        'collision_count_mean': np.mean(collision_counts),
        'collision_count_std': np.std(collision_counts),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
    }


def run_batch_multiple_seeds(
    comm_enabled: bool,
    broadcast_interval_steps: int,
    comm_range: float,
    msg_length: int,
    n_agents: int = DEFAULT_NUM_AGENTS,
    num_seeds: int = 125,
    verbose: bool = False
) -> dict:
    """
    Run batch experiments across multiple seeds.
    
    Runs num_seeds independent simulations with consecutive seeds (0, 1, 2, ..., num_seeds-1).
    Each seed produces one simulation result.
    
    Args:
        comm_enabled: Whether agents communicate
        broadcast_interval_steps: Steps between broadcasts (smaller = more frequent)
        comm_range: Communication range (px)
        msg_length: Message length (waypoints)
        n_agents: Number of agents
        num_seeds: Total number of independent samples
        verbose: Print progress messages
        
    Returns:
        dict: Aggregated statistics across all seeds (mean and std)
    """
    all_results = []
    
    for seed in range(num_seeds):
        result = run_simulation(
            comm_enabled=comm_enabled,
            broadcast_interval_steps=broadcast_interval_steps,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            seed=seed,
            verbose=False
        )
        
        if result is None:
            # Planning failed, skip this seed
            if verbose:
                print(f"  Seed {seed}: Initial planning failed, skipping...")
            continue
        
        all_results.append(result)
        
        if verbose and (seed + 1) % 10 == 0:
            print(f"  Completed {seed + 1}/{num_seeds} seeds...")
    
    if not all_results:
        raise RuntimeError("All seeds failed initial planning")
    
    # Aggregate all results (result is SimulationMetrics object)
    times = [r.total_time for r in all_results]
    distances = [r.total_distance() for r in all_results]
    replans = [r.replan_count for r in all_results]
    separations = [r.min_separation for r in all_results]
    avg_separations = [r.avg_separation for r in all_results]
    collision_rates = [1 if r.collision_occurred else 0 for r in all_results]
    collision_counts = [r.collision_count for r in all_results]
    costs = [r.compute_cost() for r in all_results]

    return {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances),
        'replan_mean': np.mean(replans),
        'replan_std': np.std(replans),
        'min_separation_mean': np.mean(separations),
        'min_separation_std': np.std(separations),
        'avg_separation_mean': np.mean(avg_separations),
        'avg_separation_std': np.std(avg_separations),
        'collision_rate': np.mean(collision_rates),
        'collision_rate_std': np.std(collision_rates),
        'collision_count_mean': np.mean(collision_counts),
        'collision_count_std': np.std(collision_counts),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
    }