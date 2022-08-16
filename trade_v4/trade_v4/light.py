import numpy as np

FIRE_LIGHT_LEVEL:     float = 0.1
MAX_LIGHT_LEVEL:      float = 1.0
STARTING_LIGHT_LEVEL: float = 0.0

def isclose(a,b):
    return abs(a-b) <= 1e-09

class Light:
    def __init__(self, grid_size, interval):
        self.gx, self.gy = self.grid_size = grid_size
        self.cx, self.cy = self.gx // 2, self.gy // 2
        self.light_level = 0
        self.interval = interval
        self.increasing = True
        self.frame = self.fire_frame()

    def reset(self):
        self.light_level = STARTING_LIGHT_LEVEL
        self.increasing = True
        self.frame = self.fire_frame()

    def dawn(self):
        return self.increasing and isclose(self.light_level, 0)

    def contains(self, pos):
        return self.frame[pos] >= 0

    def fire_frame(self):
        if self.light_level >= 0:
            return np.full(self.grid_size, self.light_level)
        fire_light = np.zeros((self.gx, self.gy))
        i, j = self.cx-3, self.cy-3
        while i <= self.cx and j <= self.cy:
            fire_light[i:self.gx-i, j:self.gy-j] += 2*self.interval
            i += 1
            j += 1
        frame = np.full(self.grid_size, self.light_level)
        frame[fire_light > 0] = np.clip(fire_light[fire_light > 0] - (self.interval * 7), self.light_level, 1)
        return frame

    def step_light(self):
        if self.increasing:
            self.light_level += self.interval
        else:
            self.light_level -= self.interval
        if isclose(abs(self.light_level), MAX_LIGHT_LEVEL):
            self.increasing = not self.increasing
        self.frame = self.fire_frame()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    l = Light((11,11), 0.2)
    l.increasing = False
    for i in range(10):
        print(l.frame.round(2))
        plt.matshow(l.frame, vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        plt.close()
        l.step_light()
