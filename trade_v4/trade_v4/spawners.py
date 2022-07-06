from abc import ABCMeta, abstractmethod
import numpy as np
import math
from .utils import two_combos, valid_pos
from random import shuffle, choice, randint, random
from itertools import product

FIRE_LIGHT_LEVEL = 0.1
MAX_LIGHT_LEVEL = 1
STARTING_LIGHT_LEVEL = 0

class BaseSpawnGenerator(metaclass=ABCMeta):
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size


    @abstractmethod
    def gen_poses(self, n=4):
        pass

class RandomSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.gx, self.gy = grid_size
    def gen_poses(self, n=4):
        return [(randint(0, self.gx-1), randint(0,self.gy-1)) for j in range(4) for i in range(4)]

class CenterSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.gx, self.gy = grid_size
        cx = (self.gx // 2)
        cy = (self.gy // 2)
        self.poses = [(cx-1, cy-1), (cx-1, cy+1), (cx+1,cy-1), (cx+1, cy+1)]
        self.reset()
        # x_poses = [cx-1, cx, cx+1]
        # y_poses = [cy-1, cy, cy+1]
        # self.poses = list(filter(lambda pos: valid_pos(pos, self.grid_size), product(x_poses, y_poses)))

    def reset(self):
        shuffle(self.poses)

    def gen_poses(self, n=4):
        return self.poses

class FourCornerSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size
        self.spawn_spots = [[(0,1,2), (0, 1,2)], [(0,1,2), (self.gy-3, self.gy-2, self.gy-1)], [(self.gx-3, self.gx-2, self.gx-1), (0,1,2)], [(self.gx-3, self.gx-2,self.gx-1), (self.gy-3,self.gy-2,self.gy-1)]]
        self.spawn_spots = [two_combos(xs, ys) for (xs, ys) in self.spawn_spots]
        self.reset()

    def reset(self):
        shuffle(self.spawn_spots)

    def gen_poses(self):
        return [choice(spot) for spot in self.spawn_spots]


class FilledCornerSpawner(BaseSpawnGenerator):
    # Like FourCornerSpawner, but spawns occasionally in a filled manner
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size
        self.n = (min(grid_size) // 2)
        self.corners = [(0,0), (0, self.gy-1), (self.gx-1, 0), (self.gx-1, self.gy-1)]
        self.reset()
        self.prob_corner = 0.8

    def reset(self):
        shuffle(self.corners)

    def sample_corner_point(self, corner):
        if random() < self.prob_corner:
            return corner
        point = []
        for c in corner:
            if c == 0:
                point.append(c + randint(0, self.n-1))
            else:
                point.append(c - randint(0, self.n-1))
        return tuple(point)

    def gen_poses(self):
        return [self.sample_corner_point(corner) for corner in self.corners]
