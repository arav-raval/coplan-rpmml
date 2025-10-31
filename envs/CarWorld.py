import pymunk
import numpy as np
import random

class CarWorld:
    def __init__(self, width=800, height=600, dt=0.05, seed=123):
        # Set seed for reproducibility
        self.seed = seed
        random.seed(self.seed)

        # Create world parameters
        self.width = width
        self.height = height
        self.dt = dt

        # Create pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        # Create agents and goals
        self.agents = []
        self.goals = []

        # Add boundaries to the world
        self._add_boundaries()

    def _add_boundaries(self, thickness=10):
        """Add walls around the map edges."""
        w, h = self.width, self.height
        static_body = self.space.static_body
        walls = [
            pymunk.Segment(static_body, (0, 0), (w, 0), thickness),   # bottom
            pymunk.Segment(static_body, (0, h), (w, h), thickness),   # top
            pymunk.Segment(static_body, (0, 0), (0, h), thickness),   # left
            pymunk.Segment(static_body, (w, 0), (w, h), thickness),   # right
        ]
        for wall in walls:
            wall.elasticity = 0.0
            wall.friction = 0.5
            self.space.add(wall)

    def add_agent(self, pos, radius=15, mass=1.0, color=(0, 0, 255)):
        """Add a circular agent to the simulation."""
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.0
        shape.friction = 0.5
        shape.color = color
        self.space.add(body, shape)

        agent = {
            "body": body,
            "shape": shape,
            "radius": radius,
            "color": color,
        }
        self.agents.append(agent)
        return agent

    def set_goal(self, pos):
        """Assign a goal coordinate (for visualization + distance calc)."""
        self.goals.append(np.array(pos, dtype=float))

    def step(self):
        self.space.step(self.dt)

    def get_positions(self):
        """Return positions of all agents."""
        return [np.array(agent["body"].position) for agent in self.agents]

    def reset(self):
        """Reset all agents to random positions."""
        for agent in self.agents:
            agent["body"].position = (
                random.uniform(50, self.width - 50),
                random.uniform(50, self.height - 50),
            )
            agent["body"].velocity = (0, 0)
