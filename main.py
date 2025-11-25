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
    PREDICTION_HORIZON, PREDICTION_DT, COLLISION_DISTANCE,
    COMMUNICATION_TRIGGER_DISTANCE, REPLAN_COOLDOWN_TIME,
    DEFAULT_NUM_AGENTS, AGENT_COLORS
)
from src.world import create_space, create_boundaries
from src.agents import Agent


def generate_random_positions(n_agents, width, height, margin=50, seed=None):
    """
    Generate random start and goal positions for agents.
    Ensures minimum separation between all positions.
    
    Returns:
        List of (start_pos, goal_pos) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    min_separation = 80  # Minimum distance between any two positions
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
            if dist > 200:  # Minimum travel distance
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


def main():
    parser = argparse.ArgumentParser(description="Multi-agent RRT simulation")
    parser.add_argument('--comm', action='store_true', help='Enable communication')
    parser.add_argument('--no-comm', dest='comm', action='store_false', help='Disable communication')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument('--agents', type=int, default=DEFAULT_NUM_AGENTS, help='Number of agents')
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

    # Initial planning
    print(f"\n=== {args.agents} Agents | Mode: {'COMMUNICATION' if args.comm else 'NO COMM'} | Seed: {args.seed} ===\n")
    for i, agent in enumerate(agents):
        print(f"Agent {i} ({AGENT_COLORS[i % len(AGENT_COLORS)]}): {positions[i][0]} → {positions[i][1]}")
        agent.plan_path()

    # Per-agent cooldowns
    replan_cooldowns = {i: 0.0 for i in range(len(agents))}
    replan_count = 0

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = 1.0 / 60.0
        space.step(dt)

        # Update cooldowns
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= dt

        # Communication-based replanning (check ALL pairs)
        if args.comm:
            for i, j in combinations(range(len(agents)), 2):
                agent_i = agents[i]
                agent_j = agents[j]
                
                # Skip if either agent is done
                if (agent_i.path is None or agent_i.path_index >= len(agent_i.path) or
                    agent_j.path is None or agent_j.path_index >= len(agent_j.path)):
                    continue
                
                pos_i = agent_i.body.position
                pos_j = agent_j.body.position
                distance = pos_i.get_distance(pos_j)

                # Check if within communication range
                if distance < COMMUNICATION_TRIGGER_DISTANCE:
                    # Check if collision predicted
                    if predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON):
                        # Lower-indexed agent has priority, higher-indexed replans
                        replanner = agent_j
                        priority_agent = agent_i
                        replanner_idx = j
                        
                        if replan_cooldowns[replanner_idx] <= 0:
                            print(f"[Collision predicted] Agent {i} vs {j} ({distance:.0f}px) → Agent {j} replans")
                            
                            # Gather all other agents' paths as obstacles
                            obstacles = []
                            for k, other in enumerate(agents):
                                if k != replanner_idx and other.path:
                                    remaining = other.get_remaining_path()
                                    if remaining:
                                        obstacles.append(remaining)
                            
                            if obstacles:
                                success = replanner.replan(dynamic_obstacles=obstacles)
                                if success:
                                    replan_count += 1
                            
                            replan_cooldowns[replanner_idx] = REPLAN_COOLDOWN_TIME

        # Update agents
        for agent in agents:
            agent.update()

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
        stats = f"Replans: {replan_count}"
        text = font.render(stats, True, (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print(f"\n=== Finished | Total replans: {replan_count} ===")


if __name__ == "__main__":
    main()