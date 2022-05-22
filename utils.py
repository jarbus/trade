directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def add_tup(tup1: tuple, tup2: tuple):
    return tuple(map(lambda i, j: i + j, tup1, tup2))

def valid_pos(pos: tuple, grid_size: tuple):
    return all(0<=p<G for (p, G) in zip(pos, grid_size))
