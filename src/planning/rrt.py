"""RRT (Rapidly-exploring Random Tree) path planner."""

import pymunk
import random
from src.config import RRT_ITERATIONS, RRT_STEP_SIZE, RRT_GOAL_BIAS, AVOIDANCE_RADIUS

class RRTPlanner:
    """RRT planner that avoids static walls and optionally dynamic paths."""

    class Node:
        """A node in the RRT tree."""
        def __init__(self, pos):
            self.pos = pymunk.Vec2d(*pos)
            self.parent = None

    def __init__(self, start_pos, goal_pos, static_obstacles, bounds):
        """
        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            static_obstacles: List of pymunk shapes to avoid (walls)
            bounds: (width, height) of the world
        """
        self.start_node = self.Node(start_pos)
        self.goal_node = self.Node(goal_pos)
        self.nodes = [self.start_node]
        self.static_obstacles = static_obstacles
        self.bounds = bounds
        self.dynamic_obstacles = []  # Paths from other agents

    def set_dynamic_obstacles(self, paths):
        """Set list of paths to avoid (for communication mode).
        
        Args:
            paths: List of paths, where each path is a list of Vec2d points.
        """
        self.dynamic_obstacles = paths

    def plan(self):
        """Generate a path from start to goal.
        
        Returns:
            List of pymunk.Vec2d waypoints, or None if no path found.
        """
        for _ in range(RRT_ITERATIONS):
            # Sample random point (with goal bias)
            if random.random() < RRT_GOAL_BIAS:
                rand_pos = self.goal_node.pos
            else:
                rand_pos = (random.randint(0, self.bounds[0]),
                            random.randint(0, self.bounds[1]))

            # Find nearest node in tree
            nearest_node = min(
                self.nodes, 
                key=lambda n: (n.pos - rand_pos).length_squared
            )

            # Steer towards random point
            dir_vec = (rand_pos - nearest_node.pos)
            if dir_vec.length == 0:
                continue
            dir_vec = dir_vec.normalized()
            new_pos = nearest_node.pos + dir_vec * RRT_STEP_SIZE

            # Check collision
            if self._is_collision_free(nearest_node.pos, new_pos):
                new_node = self.Node(new_pos)
                new_node.parent = nearest_node
                self.nodes.append(new_node)

                # Check if goal is reachable
                if new_node.pos.get_distance(self.goal_node.pos) < RRT_STEP_SIZE:
                    if self._is_collision_free(new_node.pos, self.goal_node.pos):
                        self.goal_node.parent = new_node
                        return self._reconstruct_path()

        return None  # Failed to find path

    def _is_collision_free(self, p1, p2):
        """Check if the line segment (p1, p2) is collision-free."""
        # Check static walls
        for obs in self.static_obstacles:
            if obs.segment_query(p1, p2):
                return False

        # Check dynamic paths (other agents' communicated paths)
        for path in self.dynamic_obstacles:
            for point in path:
                line_vec = p2 - p1
                if line_vec.length_squared == 0:
                    continue
                # Find closest point on segment to the path point
                point_vec = point - p1
                t = line_vec.dot(point_vec) / line_vec.length_squared
                t = max(0, min(1, t))  # Clamp to segment
                closest = p1 + t * line_vec
                if closest.get_distance(point) < AVOIDANCE_RADIUS:
                    return False
        return True

    def _reconstruct_path(self):
        """Backtrack from goal to start to build the path."""
        path = []
        curr = self.goal_node
        while curr:
            path.append(curr.pos)
            curr = curr.parent
        return path[::-1]  # Reverse: start -> goal