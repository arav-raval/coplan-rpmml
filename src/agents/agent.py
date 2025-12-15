"""Agent with physics body and path planning."""

import pymunk
import pygame
from src.config import (
    AGENT_RADIUS, MAX_SPEED, GOAL_THRESHOLD, 
    RRT_ITERATIONS_REPLAN, REPLAN_WAIT_STEPS
)
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
        self.planner = RRTPlanner(start_pos, goal_pos, static_obstacles, world_bounds, max_iterations=None)

    def plan_path(self):
        """Run initial path planning."""
        print(f"Agent ({self.shape.color}) planning path...")
        self.path = self.planner.plan()
        if self.path:
            self.path_index = 0
            print(f"  Found path with {len(self.path)} waypoints.")
        else:
            print(f"  FAILED to find path.")

    def replan(self, dynamic_obstacles=None, max_retries=3):
        """Replan from current position, optionally avoiding other paths.
        
        Uses more RRT iterations when replanning (especially with dynamic obstacles)
        and retries with increasing iterations if initial attempt fails.
        
        Args:
            dynamic_obstacles: List of paths to avoid (from other agents).
            max_retries: Maximum number of retry attempts with more iterations.
            
        Returns:
            True if replanning succeeded, False otherwise.
        """
        current_pos = self.body.position
        
        # Try with increasing iterations (more iterations for harder cases)
        for attempt in range(max_retries + 1):
            iterations = RRT_ITERATIONS_REPLAN * (attempt + 1)  # 1000, 2000, 3000, ...
            
            new_planner = RRTPlanner(
                current_pos, self.goal,
                self.static_obstacles, self.world_bounds,
                max_iterations=iterations
            )
            if dynamic_obstacles:
                new_planner.set_dynamic_obstacles(dynamic_obstacles)

            new_path = new_planner.plan()
            if new_path:
                self.path = new_path
                self.path_index = 0
                self.planner = new_planner
                agent_id = getattr(self, 'id', '?')
                if attempt > 0:
                    print(f"  Agent {agent_id} replanned on attempt {attempt+1} ({iterations} iters): {len(self.path)} waypoints")
                else:
                    print(f"  Agent {agent_id} replanned: {len(self.path)} waypoints")
                return True
        
        agent_id = getattr(self, 'id', '?')
        print(f"  ⚠️  Agent {agent_id} replan FAILED after {max_retries+1} attempts")
        
        # Fallback: insert current position as "wait" waypoints
        if self.path and self.path_index < len(self.path):
            current_pos = self.body.position
            # Insert current position multiple times to create a waiting period
            wait_waypoints = [current_pos] * REPLAN_WAIT_STEPS
            
            # Insert wait waypoints at current position in path
            self.path = (self.path[:self.path_index] + 
                        wait_waypoints + 
                        self.path[self.path_index:])
            
            print(f"  🛑 Agent {agent_id} WAITING (inserted {len(wait_waypoints)} wait waypoints)")
            return True  # Waiting counts as successful collision avoidance
        else:
            print(f"  ⚠️  Agent {agent_id} has no path to insert wait - keeping old path")
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