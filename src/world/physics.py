"""Physics world setup using pymunk."""

import pymunk

def create_space():
    """Create and return a pymunk space with no gravity (top-down view)."""
    space = pymunk.Space()
    space.gravity = (0, 0)
    return space


def create_boundaries(space, width, height):
    """Create static walls around the simulation area.
    
    Returns:
        List of wall segments (needed for collision checking in planner).
    """
    static_body = space.static_body
    walls = [
        pymunk.Segment(static_body, (0, 0), (width, 0), 2),       # bottom
        pymunk.Segment(static_body, (0, 0), (0, height), 2),      # left
        pymunk.Segment(static_body, (width, 0), (width, height), 2),  # right
        pymunk.Segment(static_body, (0, height), (width, height), 2), # top
    ]
    for w in walls:
        w.elasticity = 0.5
    space.add(*walls)
    return walls