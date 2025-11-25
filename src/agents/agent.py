"""Agent with physics body and path planning."""

import pymunk
import pygame
from src.config import AGENT_RADIUS, MAX_SPEED, GOAL_THRESHOLD
from src.planning import RRTPlanner


class Agent:
    """An agent that can plan and follow paths."""

    def __init__(self, space, start_pos, goal_pos, static_obstacles, world_bounds, color="blue"):
        """
        Args:
            space: pymunk.Space to add the agent to
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            static_obstacles: List of wall segments for planning
            world_bounds: (width, height) of the world
            color: Pygame color name
        """
        # Physics body
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, AGENT_RADIUS)
        self.body = pymunk.Body(mass, moment)
        self.body.position = start_pos
        self.shape = pymunk.Circle(self.body, AGENT_RADIUS)
        self.shape.elasticity = 0.8
        self.shape.color = pygame.color.THECOLORS[color]
        space.add(self.body, self.shape)

        # Planning state
        self.goal = pymunk.Vec2d(*goal_pos)
        self.path = None
        self.path_index = 0
        self.static_obstacles = static_obstacles
        self.world_bounds = world_bounds
        self.planner = RRTPlanner(start_pos, goal_pos, static_obstacles, world_bounds)

    def plan_path(self):
        """Run initial path planning."""
        print(f"Agent ({self.shape.color}) planning path...")
        self.path = self.planner.plan()
        if self.path:
            self.path_index = 0
            print(f"  Found path with {len(self.path)} waypoints.")
        else:
            print(f"  FAILED to find path.")

    def replan(self, dynamic_obstacles=None):
        """Replan from current position, optionally avoiding other paths.
        
        Args:
            dynamic_obstacles: List of paths to avoid (from other agents).
            
        Returns:
            True if replanning succeeded, False otherwise.
        """
        current_pos = self.body.position
        new_planner = RRTPlanner(
            current_pos, self.goal,
            self.static_obstacles, self.world_bounds
        )
        if dynamic_obstacles:
            new_planner.set_dynamic_obstacles(dynamic_obstacles)

        new_path = new_planner.plan()
        if new_path:
            self.path = new_path
            self.path_index = 0
            self.planner = new_planner
            print(f"Agent ({self.shape.color}) replanned: {len(self.path)} waypoints.")
            return True
        else:
            print(f"Agent ({self.shape.color}) replan FAILED, keeping old path.")
            return False

    def update(self):
        """Follow the planned path (call each frame)."""
        if not self.path or self.path_index >= len(self.path):
            self.body.velocity = (0, 0)
            return

        target = self.path[self.path_index]
        pos = self.body.position
        vec_to_target = target - pos
        dist = vec_to_target.length

        if dist < GOAL_THRESHOLD:
            self.path_index += 1
            if self.path_index >= len(self.path):
                print(f"Agent ({self.shape.color}) reached goal!")
                self.body.velocity = (0, 0)
        else:
            direction = vec_to_target.normalized()
            self.body.velocity = direction * MAX_SPEED

    def get_remaining_path(self):
        """Return the portion of path not yet traversed."""
        if self.path and self.path_index < len(self.path):
            return self.path[self.path_index:]
        return []