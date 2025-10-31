import math

class PurePursuit:
    def __init__(self, lookahead=25.0, v_des=100.0, k_w=3.0, stop_dist=25.0):
        self.look = lookahead
        self.v_des = v_des
        self.k_w = k_w
        self.stop_dist = stop_dist
        self.i = 0

    def reset(self): 
        self.i = 0

    def control(self, pose, path):
        if path is None or len(path) < 2:
            return 0.0, 0.0
        x, y, th = pose

        # Choose target waypoint ahead of current index
        while self.i < len(path)-1 and math.hypot(path[self.i][0]-x, path[self.i][1]-y) < self.look:
            self.i += 1
        tgt = path[min(self.i, len(path)-1)]
        dx, dy = tgt[0]-x, tgt[1]-y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        e_theta = (ang - th + math.pi)%(2*math.pi) - math.pi

        # Slow down when close to goal
        if d < self.stop_dist:
            v = self.v_des * (d / self.stop_dist)
        else:
            v = self.v_des
        if d < 5.0:
            v = 0.0
        w = self.k_w * e_theta
        return v, w
