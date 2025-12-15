# main.py
"""Entry point for the multi-agent simulation."""

import pygame
import pymunk
import pymunk.pygame_util
import argparse
import random
from itertools import combinations

from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MAX_SPEED, AGENT_RADIUS,
    PREDICTION_HORIZON, PREDICTION_DT, COLLISION_DISTANCE, EXPERIMENT_CONFIG,
    DEFAULT_NUM_AGENTS, AGENT_COLORS,
    POSITION_MARGIN, MIN_POSITION_SEPARATION, MIN_TRAVEL_DISTANCE,
    MAX_MSG_LENGTH_STEPS, SIMULATION_FPS, REPLAN_COOLDOWN_STEPS
)
from src.world import create_space, create_boundaries
from src.agents import Agent


def generate_random_positions(n_agents, width, height, margin=None, seed=None):
    """
    Generate random start and goal positions for agents.
    Ensures minimum separation between all positions.
    
    Args:
        n_agents: Number of agents
        width: Screen width
        height: Screen height
        margin: Margin from edges (defaults to POSITION_MARGIN from config)
        seed: Random seed
    
    Returns:
        List of (start_pos, goal_pos) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    if margin is None:
        margin = POSITION_MARGIN
    min_separation = MIN_POSITION_SEPARATION
    positions = []
    
    for _ in range(n_agents):
        # Generate start position
        attempts = 0
        while attempts < 100:
            start = (random.randint(margin, width - margin),
                     random.randint(margin, height - margin))
            # Check separation from existing positions
            valid = True
            for existing_start, existing_goal in positions:
                if (abs(start[0] - existing_start[0]) < min_separation and
                    abs(start[1] - existing_start[1]) < min_separation):
                    valid = False
                    break
            if valid:
                break
            attempts += 1
        
        # Generate goal position (prefer opposite side of screen)
        attempts = 0
        while attempts < 100:
            # Bias goal to opposite side
            if start[0] < width // 2:
                goal_x = random.randint(width // 2, width - margin)
            else:
                goal_x = random.randint(margin, width // 2)
            if start[1] < height // 2:
                goal_y = random.randint(height // 2, height - margin)
            else:
                goal_y = random.randint(margin, height // 2)
            goal = (goal_x, goal_y)
            
            # Ensure goal is far enough from start
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
        # Simulate agent1 movement
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

        # Simulate agent2 movement
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

        # Check for collision (must happen AFTER movement simulation)
        distance = sim_pos1.get_distance(sim_pos2)
        if distance < COLLISION_DISTANCE:
            return True

        # Early exit: if both agents have finished their paths, no collision possible
        if sim_idx1 >= len(agent1.path) and sim_idx2 >= len(agent2.path):
            return False

        t += PREDICTION_DT

    return False


def main():
    parser = argparse.ArgumentParser(description="Multi-agent RRT simulation")
    # All defaults read from EXPERIMENT_CONFIG in config.py (single source of truth)
    parser.add_argument('--comm', action='store_true', help='Enable communication')
    parser.add_argument('--no-comm', dest='comm', action='store_false', help='Disable communication')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument('--agents', type=int, default=EXPERIMENT_CONFIG['n_agents'], 
                        help=f'Number of agents (default={EXPERIMENT_CONFIG["n_agents"]})')
    parser.add_argument('--frequency', type=float, default=EXPERIMENT_CONFIG['default_frequency_hz'], 
                        help=f'Broadcast frequency (Hz, default={EXPERIMENT_CONFIG["default_frequency_hz"]}Hz)')
    parser.add_argument('--range', type=float, default=EXPERIMENT_CONFIG['comm_range'], 
                        help=f'Communication range (px, default={EXPERIMENT_CONFIG["comm_range"]:.1f}px = prediction distance)')
    parser.add_argument('--msg-length', type=int, default=EXPERIMENT_CONFIG['msg_length'], 
                        help=f'Message length (waypoints, default={EXPERIMENT_CONFIG["msg_length"]})')
    parser.set_defaults(comm=True)
    args = parser.parse_args()

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    title = f"{args.agents} Agents - {'Comm ON' if args.comm else 'Comm OFF'} - Seed {args.seed}"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()

    # Pymunk setup
    space = create_space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)

    # Generate random positions
    positions = generate_random_positions(
        args.agents, SCREEN_WIDTH, SCREEN_HEIGHT, seed=args.seed
    )

    # Create agents
    agents = []
    for i, (start, goal) in enumerate(positions):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent = Agent(space, start, goal, static_obstacles,
                      (SCREEN_WIDTH, SCREEN_HEIGHT), color)
        agent.id = i  # Add ID for tracking
        agents.append(agent)

    # Convert frequency from Hz to steps
    broadcast_interval_steps = int(SIMULATION_FPS / args.frequency) if args.frequency > 0 else float('inf')
    
    # Initial planning
    print(f"\n=== {args.agents} Agents | Mode: {'COMMUNICATION' if args.comm else 'NO COMM'} | Seed: {args.seed} ===")
    if args.comm:
        print(f"Comm params: freq={args.frequency:.2f}Hz ({broadcast_interval_steps} steps), range={args.range:.0f}px, msg_len={args.msg_length}\n")
    else:
        print()
    for i, agent in enumerate(agents):
        print(f"Agent {i} ({AGENT_COLORS[i % len(AGENT_COLORS)]}): {positions[i][0]} → {positions[i][1]}")
        agent.plan_path()

    # Per-agent cooldowns (in steps)
    replan_cooldowns = {i: 0 for i in range(len(agents))}
    replan_count = 0
    
    # Collision tracking
    collision_count = 0
    collision_pairs = set()  # Track unique collisions
    
    # Track last communication step for each agent pair
    last_comm_step = {}  # (i, j) -> step_number
    
    # Debug counters
    frame_count = 0
    last_debug_output = 0

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = 1.0 / SIMULATION_FPS
        space.step(dt)

        # Update cooldowns (decrement by 1 step)
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= 1

        # Communication-based replanning (check ALL pairs)
        if args.comm:
            # Debug: track how many pairs are in range and how many predict collisions
            pairs_in_range = 0
            pairs_predicting_collision = 0
            pairs_blocked_by_cooldown = 0
            
            for i, j in combinations(range(len(agents)), 2):
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
                last_step = last_comm_step.get(pair, -broadcast_interval_steps)  # Allow communication on first check
                
                # Bypass timing check if one agent is done (immediate notification needed)
                timing_ok = (frame_count - last_step) >= broadcast_interval_steps
                immediate_check_needed = agent_i_done or agent_j_done  # Stable agent = immediate check
                
                # Communicate if timing ok OR if immediate check needed
                if timing_ok or immediate_check_needed:
                    pos_i = agent_i.body.position
                    pos_j = agent_j.body.position
                    distance = pos_i.get_distance(pos_j)

                    # Check if within communication range (using parameter, not hardcoded)
                    if distance < args.range:
                        # Update last communication time for this pair
                        last_comm_step[pair] = frame_count
                        
                        pairs_in_range += 1
                        # Check if collision predicted
                        collision_predicted = predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON)
                        # Debug: log when agents are close but no collision predicted
                        if not collision_predicted and distance < args.range * 0.5 and frame_count % 180 == 0:  # Every 3 seconds, if within half range
                            print(f"  [Debug] Agents {i}-{j}: dist={distance:.1f}px (range={args.range:.0f}px), "
                                  f"collision_threshold={COLLISION_DISTANCE:.1f}px, "
                                  f"path_idx={agent_i.path_index}/{len(agent_i.path) if agent_i.path else 0}, "
                                  f"{agent_j.path_index}/{len(agent_j.path) if agent_j.path else 0}, "
                                  f"NO collision predicted")
                        if collision_predicted:
                            pairs_predicting_collision += 1
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
                                force_msg = " [FORCED]" if force_replan else ""
                                print(f"[Collision predicted] Agent {i} vs {j} ({distance:.0f}px) → Agent {replanner_idx} replans{force_msg}")
                            
                            # Gather paths from agents within communication range only
                            # Include done agents as stationary obstacles (single point at their position)
                            replanner_pos = replanner.body.position
                            effective_msg_length = min(args.msg_length, MAX_MSG_LENGTH_STEPS) if args.msg_length > 0 else MAX_MSG_LENGTH_STEPS
                            obstacles = []
                            total_obstacle_points = 0
                            for k, other in enumerate(agents):
                                if k != replanner_idx:
                                    # Check if this agent is within communication range
                                    other_pos = other.body.position
                                    other_distance = replanner_pos.get_distance(other_pos)
                                    if other_distance < args.range:
                                        # If agent is done, include its final position as obstacle
                                        other_done = (other.path is None or other.path_index >= len(other.path))
                                        if other_done:
                                            # Done agent broadcasts its stationary position
                                            obstacles.append([other_pos])
                                            total_obstacle_points += 1
                                        elif other.path:
                                            # Active agent broadcasts its remaining path
                                            remaining = other.get_remaining_path()
                                            if effective_msg_length > 0 and len(remaining) > effective_msg_length:
                                                remaining = remaining[:effective_msg_length]
                                            if remaining:
                                                obstacles.append(remaining)
                                                total_obstacle_points += len(remaining)
                            
                            if obstacles:
                                print(f"  [Debug] Agent {j} replanning with {len(obstacles)} obstacle paths ({total_obstacle_points} total points)")
                                success = replanner.replan(dynamic_obstacles=obstacles)
                                if success:
                                    replan_count += 1
                                    # Set cooldown from config (not tied to communication frequency)
                                    replan_cooldowns[replanner_idx] = REPLAN_COOLDOWN_STEPS
                                else:
                                    print(f"  [WARNING] Replanning failed for Agent {j} - will retry next frame")
                                    # Don't set cooldown on failure - allow immediate retry
                            else:
                                print(f"  [WARNING] No obstacles to avoid for Agent {j}")
                                # Don't set cooldown if no obstacles (shouldn't happen, but be safe)
                        else:
                            pairs_blocked_by_cooldown += 1
            
            # Debug output every 60 frames (1 second)
            frame_count += 1
            if frame_count - last_debug_output >= 60:
                if pairs_in_range > 0:
                    print(f"[Debug frame {frame_count}] Pairs in range: {pairs_in_range}, "
                          f"Predicting collision: {pairs_predicting_collision}, "
                          f"Blocked by cooldown: {pairs_blocked_by_cooldown}")
                    last_debug_output = frame_count

        # Update agents
        for agent in agents:
            agent.update()

        # Check for actual collisions (physical crashes) - GROUND TRUTH
        for i, j in combinations(range(len(agents)), 2):
            agent_i = agents[i]
            agent_j = agents[j]
            
            # Skip if either agent is done
            if (agent_i.path is None or agent_i.path_index >= len(agent_i.path) or
                agent_j.path is None or agent_j.path_index >= len(agent_j.path)):
                continue
            
            # Get actual current positions from physics engine
            pos_i = agent_i.body.position
            pos_j = agent_j.body.position
            distance = pos_i.get_distance(pos_j)
            
            # Collision = agents overlap (distance < sum of radii)
            collision_threshold = AGENT_RADIUS * 2  # 30px = physical overlap
            if distance < collision_threshold:
                pair_key = tuple(sorted([i, j]))
                if pair_key not in collision_pairs:
                    collision_pairs.add(pair_key)
                    collision_count += 1
                    print(f"💥 COLLISION! Agents {i}-{j} at distance {distance:.1f}px (frame {frame_count})")

        # Drawing
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)

        for i, agent in enumerate(agents):
            # Draw goal
            pygame.draw.circle(screen, (0, 255, 0),
                               (int(agent.goal.x), int(agent.goal.y)), 5)
            # Draw path
            if agent.path:
                points = [(int(p.x), int(p.y)) for p in agent.path]
                pygame.draw.lines(screen, agent.shape.color, False, points, 2)
            # Draw agent ID
            font = pygame.font.SysFont(None, 20)
            text = font.render(str(i), True, (0, 0, 0))
            screen.blit(text, (int(agent.body.position.x) - 5, 
                               int(agent.body.position.y) - 25))

        # Draw stats
        font = pygame.font.SysFont(None, 24)
        stats = f"Replans: {replan_count} | Collisions: {collision_count}"
        text = font.render(stats, True, (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Total replans: {replan_count}")
    print(f"Total collisions: {collision_count}")
    if collision_count > 0:
        print(f"Collision pairs: {sorted(list(collision_pairs))}")
    print()


if __name__ == "__main__":
    main()