import math
from pymunk_world import PymunkWorld, CarSpec
from grid_map import ascii_to_segments
from lidar import Lidar
from rrt import rrt
from controllers import Controller

class CoPlan2DEnv:
    def __init__(self, grid_ascii, start_goals, bounds=(0,0,600,400),
                 lidar_n=24, lidar_range=220.0, dt=0.03, comms_enabled=True):
        self.world = PymunkWorld(dt=dt)
        self.bounds = bounds
        for a,b in ascii_to_segments(grid_ascii):
            self.world.add_wall_segment(a,b,thick=2.0)
        self.car_ids = []
        self.controllers = []
        self.paths = [None, None]
        self.goals = [start_goals[0][1], start_goals[1][1]]
        self.comms_enabled = comms_enabled
        self.lidar = Lidar(self.world.space, n_rays=lidar_n, max_range=lidar_range)

        # Add cars
        for (pos, theta), _goal in start_goals:
            self.car_ids.append(self.world.add_car(pos, theta, CarSpec()))
            self.controllers.append(Controller())
        self._plan_all()

    def _plan_all(self):
        for i in range(2):
            s = tuple(self.world.cars[i]["body"].position)
            g = self.goals[i]
            self.paths[i] = rrt(self.world.space, s, g, self.bounds)
            self.controllers[i].reset()

    def _intent_token(self, i):
        # Classify next local turn from path geometry
        p = self.paths[i]
        if p is None or len(p) < 3: return "straight"
        car = self.world.cars[i]["body"]
        x, y, th = car.position.x, car.position.y, car.angle
        pp = self.controllers[i].i
        j1, j2 = min(pp, len(p)-2), min(pp+1, len(p)-1)

    def _conflict(self):
        return self.world.circle_collision(0,1)

    def _yield_logic(self, intents):
        # Priority: right > straight > left
        order = {"right":2, "straight":1, "left":0}
        if not self._conflict(): return [1.0, 1.0]
        pr0, pr1 = order.get(intents[0],1), order.get(intents[1],1)
        if pr0 > pr1: return [1.0, 0.0]
        if pr1 > pr0: return [0.0, 1.0]
        return [1.0, 0.0]  # Tie to car0

    def get_obs(self, i):
        car = self.world.cars[i]["body"]
        pos, th = (car.position.x, car.position.y), car.angle
        
        # LIDAR
        ranges = self.lidar.scan(pos, th)

        # Goal distance/bearing
        dx, dy = self.goals[i][0]-pos[0], self.goals[i][1]-pos[1]
        d = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - th

        # Nearest other 
        j = 1-i
        other = self.world.cars[j]["body"].position
        ro = math.hypot(other.x-pos[0], other.y-pos[1])
        bo = math.atan2(other.y-pos[1], other.x-pos[0]) - th
        return dict(lidar=ranges, goal=(d, ((bearing+math.pi)%(2*math.pi)-math.pi)),
                    other=(ro, ((bo+math.pi)%(2*math.pi)-math.pi)))

    def step_follow_paths(self):
        # Compute intents (for comms)
        intents = [self._intent_token(0), self._intent_token(1)]
        scales = [1.0, 1.0]
        if self.comms_enabled:
            scales = self._yield_logic(intents)

        controls = []
        for i in range(2):
            car = self.world.cars[i]["body"]
            pose = (car.position.x, car.position.y, car.angle)
            v, w = self.controllers[i].control(pose, self.paths[i])
            v *= scales[i]
            controls.append((v, w))

        self.world.step_unicycle(controls)

        # Done checks
        done = []
        for i in range(2):
            car = self.world.cars[i]["body"]
            dist_goal = math.hypot(car.position.x - self.goals[i][0],
                                car.position.y - self.goals[i][1])
            if dist_goal < 15.0:
                done.append(True)
            else:
                done.append(False)
        crash = self.world.circle_collision(0,1)

        # Episode end logic
        all_done = (done[0] and done[1]) or crash or self.world.t > 30.0

        info = {"intents": intents, "crash": crash}
        return all_done, info
