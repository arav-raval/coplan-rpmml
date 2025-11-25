# main.py
"""Entry point for the multi-agent simulation."""

import pygame
import pymunk
import pymunk.pygame_util
import argparse
import random

from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MAX_SPEED,
    PREDICTION_HORIZON, PREDICTION_DT, COLLISION_DISTANCE,
    COMMUNICATION_TRIGGER_DISTANCE, REPLAN_COOLDOWN_TIME
)
from src.world import create_space, create_boundaries
from src.agents import Agent


def predict_collision(agent1, agent2, horizon_seconds):
    """Simulate both agents forward to check for future collision.
    
    Returns:
        True if collision predicted within horizon, False otherwise.
    """
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
        # Simulate agent 1
        if sim_idx1 < len(agent1.path):
            target1 = agent1.path[sim_idx1]
            vec1 = target1 - sim_pos1
            move = MAX_SPEED * PREDICTION_DT
            if move >= vec1.length:
                sim_pos1 = target1
                sim_idx1 += 1
            else:
                sim_pos1 += vec1.normalized() * move

        # Simulate agent 2
        if sim_idx2 < len(agent2.path):
            target2 = agent2.path[sim_idx2]
            vec2 = target2 - sim_pos2
            move = MAX_SPEED * PREDICTION_DT
            if move >= vec2.length:
                sim_pos2 = target2
                sim_idx2 += 1
            else:
                sim_pos2 += vec2.normalized() * move

        # Check collision
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
    parser.set_defaults(comm=True)
    args = parser.parse_args()

    random.seed(args.seed)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    title = "With Communication" if args.comm else "No Communication"
    pygame.display.set_caption(f"RRT Agents - {title}")
    clock = pygame.time.Clock()

    # Pymunk setup
    space = create_space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)

    # Create agents (opposite corners, collision course)
    agent1 = Agent(space, (100, 100), (700, 500), static_obstacles,
                   (SCREEN_WIDTH, SCREEN_HEIGHT), "blue")
    agent2 = Agent(space, (700, 450), (100, 100), static_obstacles,
                   (SCREEN_WIDTH, SCREEN_HEIGHT), "red")
    agents = [agent1, agent2]

    # Initial planning
    print(f"\n=== Mode: {'COMMUNICATION' if args.comm else 'NO COMMUNICATION'} ===\n")
    agent1.plan_path()
    agent2.plan_path()

    replan_cooldown = 0.0

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = 1.0 / 60.0
        space.step(dt)

        # Communication-based replanning
        if args.comm:
            if replan_cooldown > 0:
                replan_cooldown -= dt

            pos1 = agent1.body.position
            pos2 = agent2.body.position
            distance = pos1.get_distance(pos2)

            if distance < COMMUNICATION_TRIGGER_DISTANCE and replan_cooldown <= 0:
                if predict_collision(agent1, agent2, PREDICTION_HORIZON):
                    print(f"\n--- COLLISION PREDICTED ({distance:.1f}px apart) ---")
                    # Agent 2 replans around agent 1's remaining path
                    remaining = agent1.get_remaining_path()
                    if remaining:
                        agent2.replan(dynamic_obstacles=[remaining])
                    replan_cooldown = REPLAN_COOLDOWN_TIME

        # Update agents
        for agent in agents:
            agent.update()

        # Drawing
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)

        for agent in agents:
            # Draw goal
            pygame.draw.circle(screen, (0, 255, 0),
                               (int(agent.goal.x), int(agent.goal.y)), 5)
            # Draw path
            if agent.path:
                points = [(int(p.x), int(p.y)) for p in agent.path]
                pygame.draw.lines(screen, agent.shape.color, False, points, 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()