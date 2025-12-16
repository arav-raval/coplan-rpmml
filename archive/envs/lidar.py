# coplan2d/env/lidar.py
import math, pymunk

class Lidar:
    def __init__(self, space, n_rays=16, max_range=200.0):
        self.space = space
        self.n = n_rays
        self.rmax = max_range

    def scan(self, pos, theta, include_shapes=None):
        """Returns ranges in [0, rmax]. include_shapes lets you treat other cars as obstacles."""
        ranges = []
        for k in range(self.n):
            ang = theta + (2*math.pi)*k/self.n
            dx, dy = math.cos(ang)*self.rmax, math.sin(ang)*self.rmax
            hit = self.space.segment_query_first(pos, (pos[0]+dx, pos[1]+dy), 0,
                                                 pymunk.ShapeFilter())
            if hit is None:
                ranges.append(self.rmax)
            else:
                ranges.append(hit.alpha * self.rmax)
        return ranges
