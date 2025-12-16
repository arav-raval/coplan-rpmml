import random, math, pymunk
from collections import namedtuple
Node = namedtuple("Node", "x y parent")

def dist(a, b): 
    return math.hypot(a[0]-b[0], a[1]-b[1])

def collision_free(space, p, q, step=5.0):
    # Sample along segment and check shapes via point query
    L = dist(p, q)
    n = max(1, int(L/step))
    for i in range(n+1):
        x = p[0] + (q[0]-p[0])*i/n
        y = p[1] + (q[1]-p[1])*i/n
        # Use segment query micro-step to detect wall proximity
        hit = space.segment_query_first((x,y), (x+1e-3, y+1e-3), 0, pymunk.ShapeFilter())
        if hit is not None and not hit.shape.body.body_type == 0:  # Static walls are fine to detect
            return False
    return True

def rrt(space, start, goal, bounds, iters=1500, step_size=25.0, goal_bias=0.1, goal_thresh=20.0):
    nodes = [Node(start[0], start[1], -1)]
    for t in range(iters):
        if random.random() < goal_bias:
            sx, sy = goal
        else:
            sx = random.uniform(bounds[0], bounds[2])
            sy = random.uniform(bounds[1], bounds[3])

        # Nearest
        idx = min(range(len(nodes)), key=lambda i: (nodes[i].x - sx)**2 + (nodes[i].y - sy)**2)
        nx, ny = nodes[idx].x, nodes[idx].y
        ang = math.atan2(sy-ny, sx-nx)
        px, py = nx + step_size*math.cos(ang), ny + step_size*math.sin(ang)

        if not collision_free(space, (nx, ny), (px, py)): 
            continue
        nodes.append(Node(px, py, idx))

        if dist((px, py), goal) < goal_thresh and collision_free(space, (px, py), goal):
            # Reconstruct
            path = [goal]
            j = len(nodes)-1
            while j >= 0:
                path.append((nodes[j].x, nodes[j].y))
                j = nodes[j].parent
            return path[::-1]
    return None
