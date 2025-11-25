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
    COMMUNICATION_TRIGGER_DISTANCE
)
from src.world import create_space, create_boundaries
from src.agents import Agent
from src.metrics import MetricsCollector, SimulationMetrics


def generate_random_positions(n_agents, width, height, margin=50):
    """Generate random start and goal positions for agents."""
    min_separation = 80
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
            if dist > 200:
                break
            attempts += 1
        
        positions.append((start, goal))
    
    return positions


def predict_collision_pair(agent1, agent2, horizon_seconds):
    """Check if two agents will collide."""
    if not agent1.path or agent1.path_index >= len(agent1.path):
        return False
    if not agent2.path or agent2.path_index >= len(agent2.path):
        return False

    sim_pos1 = pymunk.Vec2d(*agent1.body.position)
    sim_idx1 = agent1.path_index
    sim_pos2 = pymunk.Vec2d(*agent2.body.position)
    sim_idx2 = agent2.path_index

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
    broadcast_frequency: float = 0.67,
    comm_range: float = 250.0,
    msg_length: int = 10,
    n_agents: int = DEFAULT_NUM_AGENTS,
    seed: int = 5,
    max_time: float = 30.0,
    verbose: bool = False
) -> SimulationMetrics:
    """
    Run a single headless simulation with N agents.
    
    Args:
        comm_enabled: Whether agents communicate
        broadcast_frequency: How often agents can replan (Hz)
        comm_range: Distance at which agents can communicate
        msg_length: Number of waypoints to share per message
        n_agents: Number of agents in simulation
        seed: Random seed for reproducibility
        max_time: Maximum simulation time before timeout
        verbose: Print status messages
        
    Returns:
        SimulationMetrics with all collected data
    """
    random.seed(seed)
    
    replan_cooldown_time = 1.0 / broadcast_frequency if broadcast_frequency > 0 else float('inf')
    
    # Setup world
    space = create_space()
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Generate random positions and create agents
    positions = generate_random_positions(n_agents, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    agents = []
    for i, (start, goal) in enumerate(positions):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent = Agent(space, start, goal, static_obstacles,
                      (SCREEN_WIDTH, SCREEN_HEIGHT), color)
        agent.id = i
        agents.append(agent)
    
    # Initial planning
    for agent in agents:
        agent.plan_path()
    
    # Check if any planning failed
    if any(agent.path is None for agent in agents):
        metrics = SimulationMetrics()
        metrics.timed_out = True
        return metrics
    
    # Metrics collection (uses first two agents for backward compatibility)
    collector = MetricsCollector(
        broadcast_frequency=broadcast_frequency,
        comm_range=comm_range,
        msg_length=msg_length
    )
    
    # Per-agent cooldowns
    replan_cooldowns = {i: 0.0 for i in range(n_agents)}
    
    # Simulation loop
    dt = 1.0 / 60.0
    current_time = 0.0
    
    while current_time < max_time:
        space.step(dt)
        current_time += dt
        
        # Update cooldowns
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= dt
        
        # Communication-based replanning (check ALL pairs)
        if comm_enabled:
            for i, j in combinations(range(n_agents), 2):
                agent_i = agents[i]
                agent_j = agents[j]
                
                # Skip if either agent is done
                if (agent_i.path is None or agent_i.path_index >= len(agent_i.path) or
                    agent_j.path is None or agent_j.path_index >= len(agent_j.path)):
                    continue
                
                pos_i = agent_i.body.position
                pos_j = agent_j.body.position
                distance = pos_i.get_distance(pos_j)

                if distance < comm_range:
                    if predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON):
                        # Higher-indexed agent replans
                        replanner_idx = j
                        replanner = agent_j
                        
                        if replan_cooldowns[replanner_idx] <= 0:
                            if verbose:
                                print(f"  [t={current_time:.2f}s] Collision: Agent {i} vs {j} → {j} replans")
                            
                            # Gather all other agents' paths as obstacles
                            obstacles = []
                            for k, other in enumerate(agents):
                                if k != replanner_idx and other.path:
                                    remaining = other.get_remaining_path()
                                    if msg_length > 0 and len(remaining) > msg_length:
                                        remaining = remaining[:msg_length]
                                    if remaining:
                                        obstacles.append(remaining)
                            
                            if obstacles:
                                success = replanner.replan(dynamic_obstacles=obstacles)
                                if success:
                                    collector.record_replan()
                            
                            replan_cooldowns[replanner_idx] = replan_cooldown_time
        
        # Update agents
        for agent in agents:
            agent.update()
        
        # Update metrics (track first two agents for compatibility)
        if len(agents) >= 2:
            collector.update(agents[0], agents[1], current_time, AGENT_RADIUS * 2)
            
            # Also track minimum separation across ALL pairs
            for i, j in combinations(range(n_agents), 2):
                sep = agents[i].body.position.get_distance(agents[j].body.position)
                if sep < collector.metrics.min_separation:
                    collector.metrics.min_separation = sep
                if sep < AGENT_RADIUS * 2:
                    collector.metrics.collision_occurred = True
        
        # Check if all agents done
        all_done = all(
            agent.path is None or agent.path_index >= len(agent.path)
            for agent in agents
        )
        if all_done:
            collector.metrics.both_reached_goal = True
            break
    
    timed_out = current_time >= max_time and not collector.metrics.both_reached_goal
    return collector.finalize(timed_out=timed_out)


def run_batch(
    comm_enabled: bool,
    broadcast_frequency: float,
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
        metrics = run_simulation(
            comm_enabled=comm_enabled,
            broadcast_frequency=broadcast_frequency,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
            seed=seed,
            verbose=verbose
        )
        results.append(metrics)
    
    # Aggregate
    times = [m.total_time for m in results]
    distances = [m.total_distance() for m in results]
    replans = [m.replan_count for m in results]
    separations = [m.min_separation for m in results]
    collisions = [1 if m.collision_occurred else 0 for m in results]
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
        'collision_rate': np.mean(collisions),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
    }