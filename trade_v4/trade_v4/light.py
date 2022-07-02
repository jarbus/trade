import numpy as np

FIRE_LIGHT_LEVEL:     float = 0.1
MAX_LIGHT_LEVEL:      float = 1.0
STARTING_LIGHT_LEVEL: float = 0.0

def isclose(a,b):
    return abs(a-b) <= 1e-09

class Light:
    def __init__(self, grid_size, interval):
        gx, gy = self.grid_size = grid_size
        tx, ty = tuple([(gx//2)-1, ((gx//2)+2)]), tuple([(gy//2)-1, ((gy//2)+2)])
        self.fire_slice = slice(*tx), slice(*ty)
        self.fire_range = range(*tx), range(*ty)
        self.light_level = 0
        self.interval = interval
        self.increasing = True

    def reset(self):
        self.light_level = STARTING_LIGHT_LEVEL
        self.increasing = True

    def dawn(self):
        return self.increasing and isclose(self.light_level, 0)

    def contains(self, pos):
        if self.light_level >= 0:
            return True
        if pos[0] in self.fire_range[0] and pos[1] in self.fire_range[1]:
            return True
        return False

    def frame(self):
        frame = np.full(self.grid_size, self.light_level)
        frame[self.fire_slice[0], self.fire_slice[1]] = FIRE_LIGHT_LEVEL
        return frame

    def step_light(self):
        if self.increasing:
            self.light_level += self.interval
        else:
            self.light_level -= self.interval
        if isclose(abs(self.light_level), MAX_LIGHT_LEVEL):
            self.increasing = not self.increasing
