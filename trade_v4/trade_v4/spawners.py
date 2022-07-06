from abc import ABCMeta, abstractmethod
import numpy as np
import math
from .utils import two_combos, valid_pos
from random import shuffle, choice, randint
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
        # x_poses = [cx-1, cx, cx+1]
        # y_poses = [cy-1, cy, cy+1]
        # self.poses = list(filter(lambda pos: valid_pos(pos, self.grid_size), product(x_poses, y_poses)))

    def gen_poses(self, n=4):
        shuffle(self.poses)
        return self.poses

class FourCornerSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size
        self.spawn_spots = [[(0,1,2), (0, 1,2)], [(0,1,2), (self.gy-3, self.gy-2, self.gy-1)], [(self.gx-3, self.gx-2, self.gx-1), (0,1,2)], [(self.gx-3, self.gx-2,self.gx-1), (self.gy-3,self.gy-2,self.gy-1)]]
        self.spawn_spots = [two_combos(xs, ys) for (xs, ys) in self.spawn_spots]
        shuffle(self.spawn_spots)

    def gen_poses(self):
        return [choice(spot) for spot in self.spawn_spots]
