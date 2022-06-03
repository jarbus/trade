import math
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def add_tup(tup1: tuple, tup2: tuple):
    return tuple(map(lambda i, j: i + j, tup1, tup2))

def valid_pos(pos: tuple, grid_size: tuple):
    return all(0<=p<G for (p, G) in zip(pos, grid_size))

def inv_dist(tup1: tuple, tup2: tuple):
    return 1/(1+math.sqrt(sum((j-i)**2 for i,j in zip(tup1, tup2))))

def two_combos(xs: tuple, ys: tuple):
    return [(x, y) for x in xs for y in ys]
