"""
Message Length Sweep Experiment

Tests how message length (number of waypoints shared) affects:
- Collision avoidance effectiveness
- Communication overhead
- Overall system cost

Message length controls information sharing:
- 0 = unlimited (entire remaining path)
- Low values = less bandwidth, potentially less collision avoidance
- High values = more bandwidth, potentially better collision avoidance
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from itertools import combinations

from src.config import (
    EXPERIMENT_CONFIG, NUM_SEEDS, NUM_SEEDS_QUICK, AGENT_RADIUS, MAX_SPEED, GOAL_THRESHOLD,
    RRT_STEP_SIZE, AVOIDANCE_RADIUS, COLLISION_DISTANCE, PREDICTION_HORIZON,
    REPLAN_COOLDOWN_STEPS, MAX_MSG_LENGTH_STEPS
)
from src.simulation import run_batch_multiple_seeds
from src.visualization.plots import plot_msg_length_sweep


def visualize_single_trial(broadcast_interval_steps, comm_range, msg_length, n_agents, seed=0):
    """
    Run a single trial with visualization for message length sweep.
    
    Args:
        broadcast_interval_steps: Communication interval (fixed for msg length sweep)
        comm_range: Communication range in pixels
        msg_length: Number of waypoints to share (0 = unlimited)
        n_agents: Number of agents
        seed: Random seed for reproducibility
    """
    import random
    import pygame
    import pymunk
    from src.world.physics import create_space, create_boundaries
    from src.agents.agent import Agent
    from src.simulation import generate_random_positions, predict_collision_pair
    from src.config import SCREEN_WIDTH, SCREEN_HEIGHT, SIMULATION_FPS, AGENT_COLORS
    
    # Initialize
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Message Length Sweep - {msg_length} waypoints - Seed {seed}")
    clock = pygame.time.Clock()
    
    # Pymunk setup
    space = create_space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES
    
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Generate positions
    positions = generate_random_positions(n_agents, SCREEN_WIDTH, SCREEN_HEIGHT, seed=seed)
    
    # Create agents
    agents = []
    for i, (start, goal) in enumerate(positions):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent = Agent(space, start, goal, static_obstacles,
                      (SCREEN_WIDTH, SCREEN_HEIGHT), color)
        agent.id = i
        agents.append(agent)
    
    # Initial planning
    print(f"  Initial planning...")
    for i, agent in enumerate(agents):
        agent.plan_path()
        if agent.path:
            print(f"    Agent {i}: {len(agent.path)} waypoints")
    
    if not all(agent.path for agent in agents):
        print(f"  ⚠️  Initial planning failed for seed {seed}")
        pygame.quit()
        return
    
    # Tracking
    frame_count = 0
    replan_count = 0
    replan_cooldowns = {i: 0 for i in range(len(agents))}
    last_comm_step = {}
    collision_count = 0
    active_collision_pairs = set()
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update physics
        dt = 1.0 / SIMULATION_FPS
        space.step(dt)
        frame_count += 1
        
        # Update cooldowns
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= 1
        
        # Update agents
        all_done = True
        for agent in agents:
            if agent.path and agent.path_index < len(agent.path):
                agent.update()
                all_done = False
        
        if all_done:
            running = False
        
        # Check collisions
        for i, j in combinations(range(len(agents)), 2):
            pos_i = agents[i].body.position
            pos_j = agents[j].body.position
            distance = pos_i.get_distance(pos_j)
            
            pair = (min(i, j), max(i, j))
            if distance < AGENT_RADIUS * 2:
                if pair not in active_collision_pairs:
                    collision_count += 1
                    active_collision_pairs.add(pair)
            else:
                active_collision_pairs.discard(pair)
        
        # Communication-based replanning
        for i, j in combinations(range(len(agents)), 2):
            agent_i = agents[i]
            agent_j = agents[j]
            
            agent_i_done = (agent_i.path is None or agent_i.path_index >= len(agent_i.path))
            agent_j_done = (agent_j.path is None or agent_j.path_index >= len(agent_j.path))
            
            if agent_i_done and agent_j_done:
                continue
            
            pair = (min(i, j), max(i, j))
            last_step = last_comm_step.get(pair, -broadcast_interval_steps)
            
            timing_ok = (frame_count - last_step) >= broadcast_interval_steps
            immediate_check_needed = agent_i_done or agent_j_done
            
            if timing_ok or immediate_check_needed:
                pos_i = agent_i.body.position
                pos_j = agent_j.body.position
                distance = pos_i.get_distance(pos_j)
                
                if distance < comm_range:
                    last_comm_step[pair] = frame_count
                    
                    collector.record_message()
                    
                    if predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON):
                        force_replan = False
                        
                        if agent_i_done and not agent_j_done:
                            replanner_idx = j
                            replanner = agent_j
                            force_replan = True
                        elif agent_j_done and not agent_i_done:
                            replanner_idx = i
                            replanner = agent_i
                            force_replan = True
                        else:
                            agent_i_on_cooldown = replan_cooldowns[i] > 0
                            agent_j_on_cooldown = replan_cooldowns[j] > 0
                            
                            if agent_i_on_cooldown and not agent_j_on_cooldown:
                                replanner_idx = j
                                replanner = agent_j
                                force_replan = True
                            elif agent_j_on_cooldown and not agent_i_on_cooldown:
                                replanner_idx = i
                                replanner = agent_i
                                force_replan = True
                            else:
                                replanner_idx = j
                                replanner = agent_j
                                force_replan = False
                        
                        if force_replan or replan_cooldowns[replanner_idx] <= 0:
                            replanner_pos = replanner.body.position
                            effective_msg_length = min(msg_length, MAX_MSG_LENGTH_STEPS) if msg_length > 0 else MAX_MSG_LENGTH_STEPS
                            obstacles = []
                            
                            for k, other in enumerate(agents):
                                if k != replanner_idx:
                                    other_pos = other.body.position
                                    other_distance = replanner_pos.get_distance(other_pos)
                                    if other_distance < comm_range:
                                        other_done = (other.path is None or other.path_index >= len(other.path))
                                        if other_done:
                                            obstacles.append([other_pos])
                                        elif other.path:
                                            remaining = other.get_remaining_path()
                                            if effective_msg_length > 0 and len(remaining) > effective_msg_length:
                                                remaining = remaining[:effective_msg_length]
                                            if remaining:
                                                obstacles.append(remaining)
                            
                            if obstacles:
                                success = replanner.replan(dynamic_obstacles=obstacles)
                                if success:
                                    replan_count += 1
                                    replan_cooldowns[replanner_idx] = REPLAN_COOLDOWN_STEPS
        
        # Render
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        
        # Draw paths
        for agent in agents:
            if agent.path:
                color = agent.shape.color
                if len(color) == 4:  
                    r, g, b, a = color
                else:  
                    r, g, b = color
                
                light_color = (min(255, r + 150), min(255, g + 150), min(255, b + 150))
                bright_color = (r, g, b)
                
                for k in range(len(agent.path) - 1):
                    start = agent.path[k]
                    end = agent.path[k + 1]
                    pygame.draw.line(screen, light_color, 
                                   (int(start.x), int(start.y)), 
                                   (int(end.x), int(end.y)), 1)
                
                if agent.path_index < len(agent.path):
                    for k in range(agent.path_index, len(agent.path) - 1):
                        start = agent.path[k]
                        end = agent.path[k + 1]
                        pygame.draw.line(screen, bright_color, 
                                       (int(start.x), int(start.y)), 
                                       (int(end.x), int(end.y)), 2)
        
        for agent in agents:
            pygame.draw.circle(screen, (0, 200, 0), 
                             (int(agent.goal.x), int(agent.goal.y)), 
                             AGENT_RADIUS // 2, 2)
        
        font = pygame.font.Font(None, 24)
        collision_status = f"💥 {collision_count}" if collision_count > 0 else "✓ 0"
        info_lines = [
            f"Message Length Sweep | Seed: {seed}",
            f"Msg Length: {msg_length} wp{' (unlimited)' if msg_length == 0 else ''} | Freq: {broadcast_interval_steps} steps",
            f"Frame: {frame_count} | Replans: {replan_count} | Collisions: {collision_status}",
            f"Agents: {n_agents} | Range: {comm_range:.0f}px",
            f"Press ESC to continue to next experiment"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        small_font = pygame.font.Font(None, 18)
        for i, agent in enumerate(agents):
            status = "✓" if (agent.path is None or agent.path_index >= len(agent.path)) else "→"
            remaining = len(agent.path) - agent.path_index if agent.path and agent.path_index < len(agent.path) else 0
            cooldown = replan_cooldowns[i]
            
            status_text = f"Agent {i}: {status} {remaining} wpts"
            if cooldown > 0:
                status_text += f" (cd: {cooldown})"
            
            text_surface = small_font.render(status_text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 140 + i * 20))
        
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()


def run_msg_length_sweep(quick=False, visualize=False):
    """
    Sweep message length (number of waypoints shared per communication).
    
    Tests how much path information needs to be shared for effective collision avoidance.
    - 0 = unlimited (share entire remaining path)
    - Low values = less bandwidth, potentially less collision avoidance
    - High values = more bandwidth, potentially better collision avoidance
    """
    print("\n" + "="*75)
    print("MESSAGE LENGTH SWEEP EXPERIMENT")
    print("="*75)
    
    # Get configuration
    n_agents = EXPERIMENT_CONFIG['n_agents']
    comm_range = EXPERIMENT_CONFIG['comm_range']
    
    broadcast_interval_steps = 12  
    
    if quick:
        min_length = EXPERIMENT_CONFIG['msg_length_sweep_min']
        max_length = EXPERIMENT_CONFIG['msg_length_sweep_max']
        num_values = EXPERIMENT_CONFIG['msg_length_sweep_num_values_quick']
        num_seeds = NUM_SEEDS_QUICK
    else:
        min_length = EXPERIMENT_CONFIG['msg_length_sweep_min']
        max_length = EXPERIMENT_CONFIG['msg_length_sweep_max']
        num_values = EXPERIMENT_CONFIG['msg_length_sweep_num_values']
        num_seeds = NUM_SEEDS
    
    if min_length == 0:
        msg_lengths = [0] + list(np.linspace(5, max_length, num_values - 1, dtype=int))
    else:
        msg_lengths = np.linspace(min_length, max_length, num_values, dtype=int)
    
    msg_lengths = sorted(set(msg_lengths))  
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Communication range: {comm_range:.0f}px")
    print(f"  Broadcast interval: {broadcast_interval_steps} steps (fixed)")
    print(f"  Message lengths: {min_length} to {max_length} waypoints")
    print(f"  Testing {len(msg_lengths)} values")
    print()
    
    start_time = time.time()
    results = []
    valid_lengths = []
    
    for i, msg_len in enumerate(msg_lengths):
        print(f"[{i+1}/{len(msg_lengths)}] {msg_len:3d} waypoints", end="")
        if msg_len == 0:
            print(" (unlimited)", end="")
        print(" ... ", flush=True)
        
        if visualize:
            viz_seed = 0
            print(f"\n  Launching visualization (seed={viz_seed})...")
            print(f"  Press ESC when ready to continue to next parameter...")
            visualize_single_trial(
                broadcast_interval_steps=broadcast_interval_steps,
                comm_range=comm_range,
                msg_length=msg_len,
                n_agents=n_agents,
                seed=viz_seed
            )
            print(f"  Visualization complete. Continuing to next parameter...\n")
            continue
        
        result = run_batch_multiple_seeds(
            comm_enabled=True,
            broadcast_interval_steps=broadcast_interval_steps,
            comm_range=comm_range,
            msg_length=msg_len,
            n_agents=n_agents,
            num_seeds=num_seeds,
            verbose=False
        )
        
        print(f"cost={result['cost_mean']:6.1f}±{result['cost_std']:4.1f}, "
              f"safety={result['avg_separation_mean']:5.1f}±{result['avg_separation_std']:4.1f}px")
        
        result['msg_length'] = msg_len
        results.append(result)
        valid_lengths.append(msg_len)
    
    if not results:
        print("\n  No results collected (visualization mode only)")
        return None
    
    elapsed = time.time() - start_time
    
    # Find optimal
    costs = [r['cost_mean'] for r in results]
    opt_idx = np.argmin(costs)
    
    # Generate plots
    print("\n" + "="*75)
    print("GENERATING PLOTS...")
    print("="*75)
    
    plot_msg_length_sweep(
        msg_lengths=valid_lengths,
        results=results,
        save_path="results/msg_length_sweep.png",
        config={
            'n_agents': n_agents,
            'num_seeds': num_seeds,
            'comm_range': comm_range,
            'broadcast_interval_steps': broadcast_interval_steps,
        }
    )
    
    print(f" Saved to results/msg_length_sweep.png")
    print(f" Completed in {elapsed:.1f}s")
    
    if not quick:
        print(f"\n Optimal: {valid_lengths[opt_idx]} waypoints, cost={costs[opt_idx]:.1f}")
    
    return {
        'optimal_msg_length': valid_lengths[opt_idx],
        'optimal_cost': costs[opt_idx],
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Message Length Sweep Experiment")
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer values')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize each parameter value')
    
    args = parser.parse_args()
    
    run_msg_length_sweep(quick=args.quick, visualize=args.visualize)
