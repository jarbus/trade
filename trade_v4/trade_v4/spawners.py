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
        # x_poses = [cx-1, cx, cx+1]
        # y_poses = [cy-1, cy, cy+1]
        # self.poses = list(filter(lambda pos: valid_pos(pos, self.grid_size), product(x_poses, y_poses)))

    def reset(self):
        shuffle(self.poses)

    def gen_poses(self, n=4):
        return self.poses

class DoubleCenterSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.gx, self.gy = grid_size
        cx = (self.gx // 2)
        cy = (self.gy // 2)
        self.poses = [(cx-1, cy-1), (cx+1, cy+1)]
        # x_poses = [cx-1, cx, cx+1]
        # y_poses = [cy-1, cy, cy+1]
        # self.poses = list(filter(lambda pos: valid_pos(pos, self.grid_size), product(x_poses, y_poses)))

    def reset(self):
        shuffle(self.poses)

    def gen_poses(self):
        return self.poses


class DoubleFilledCornerSpawner(BaseSpawnGenerator):
    # Like FourCornerSpawner, but spawns occasionally in a filled manner
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size
        self.n = (min(grid_size) // 2)
        self.corners = [(0,0), (self.gx-1, self.gy-1)]
        self.init_corner_probs(self.n)

    def reset(self):
        shuffle(self.corners)

    def init_corner_probs(self, radius):
        qc = np.zeros((radius, radius))
        # generate quarter-cirl with depleting probs as radius increases
        for r in range(radius):
            for i in range(r):
                for j in range(r):
                    if (i)**2 + (j)**2 <= radius**2:
                        qc[i, j] += (radius - r)**2

        self.idxs = np.indices(qc.shape).reshape(2, -1).T
        self.probs = qc.flatten() / np.sum(qc)

    def sample_corner_offset(self):
        return self.idxs[np.random.choice(np.arange(len(self.idxs)), p=self.probs)]

    def sample_corner_point(self, corner):
        offset = self.sample_corner_offset()
        return (abs(corner[0] - offset[0]), abs(corner[1] - offset[1]))

    def gen_poses(self):
        return [self.sample_corner_point(corner) for corner in self.corners]





class FourCornerSpawner(BaseSpawnGenerator):
    def __init__(self, grid_size):
        self.gx, self.gy = grid_size
        self.spawn_spots = [[(0,1,2), (0, 1,2)], [(0,1,2), (self.gy-3, self.gy-2, self.gy-1)], [(self.gx-3, self.gx-2, self.gx-1), (0,1,2)], [(self.gx-3, self.gx-2,self.gx-1), (self.gy-3,self.gy-2,self.gy-1)]]
        self.spawn_spots = [two_combos(xs, ys) for (xs, ys) in self.spawn_spots]

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
        self.init_corner_probs(self.n)

    def reset(self):
        shuffle(self.corners)

    def init_corner_probs(self, radius):
        qc = np.zeros((radius, radius))
        # generate quarter-cirl with depleting probs as radius increases
        for r in range(radius):
            for i in range(r):
                for j in range(r):
                    if (i)**2 + (j)**2 <= radius**2:
                        qc[i, j] += (radius - r)**2

        self.idxs = np.indices(qc.shape).reshape(2, -1).T
        self.probs = qc.flatten() / np.sum(qc)

    def sample_corner_offset(self):
        return self.idxs[np.random.choice(np.arange(len(self.idxs)), p=self.probs)]

    def sample_corner_point(self, corner):
        offset = self.sample_corner_offset()
        return (abs(corner[0] - offset[0]), abs(corner[1] - offset[1]))

    def gen_poses(self):
        return [self.sample_corner_point(corner) for corner in self.corners]

if __name__ == "__main__":
    print("hi")
    size = (11,11)
    fc = DoubleFilledCornerSpawner(size)
    x = np.zeros(size)
    import matplotlib.pyplot as plt
    for i in range(100):
        for pos in fc.gen_poses():
            x[pos] += 0.1
    print(x)
    import matplotlib.pyplot as plt
    plt.matshow(x)
    plt.show()
