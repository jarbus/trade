import re
import gif
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, List



#plt.style.use('dark_background')
parser = argparse.ArgumentParser(description="Convert serve file to gif")
parser.add_argument("file", type=str)
args = parser.parse_args()

player_expr = r"(player_\d): \((\d), (\d)\) \[(.*), (.*)\] (.*)$"
exchange_expr = r"Exchange: (player_\d) gave (\S*) of food (\d) to (player_\d)"
food_expr = r"food(\d):"
light_expr = r"Light:$"

@dataclass
class Player:
    name: str
    pos: Tuple[int, int]
    food_count: Tuple[float]
    done: bool

    def __str__(self):
        fcs = [f'{food:4.1f}' for food in self.food_count]
        return f"{self.name} {','.join(fcs)} {self.done}"

@dataclass
class Step:
    idx: int
    players: List[Player]
    exchange_messages: List[str]
    food_grid: List[List[float]]
    light_grid: List[List[float]]


all_exchange_messages = []
player_colors = ["olivedrab", "darkcyan", "mediumslateblue", "purple"]
food_colors = ["red", "yellow"]
offv = 0.01
player_offsets = (0, offv), (offv, 0), (0, -offv), (-offv, 0)
food_offsets = (offv, offv), (-offv, -offv)
grid_offset = (0.01, 0.01)
def add_tuple(a, b):
    return tuple(i + j for i, j in zip(a, b))

@gif.frame
def plot_step(step: Step):

    fig = plt.figure()
    vs = 0.6
    hs = 0.6
    #axes.append(fig.add_axes([x, y, w, h]))
    grid = fig.add_axes([0.05, 0.15, vs - 0.1, 0.7])
    #grid.set_facecolor(f'{(float(step.light_level) + 1) / 2}')
    player_info = fig.add_axes([vs, hs, 1-vs, 1-hs])
    exchange_info = fig.add_axes([vs, 0, 1-vs, hs])
    scale = len(step.food_grid[0])

    for ax in [grid, player_info, exchange_info]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0, 0.95, f"Step {step.idx}", fontsize=12, wrap=True)
    for i, row in enumerate(step.light_grid[0]):
        for j, col in enumerate(row):
            l_pos = (i/scale, j/scale)
            l_pos = add_tuple(l_pos, grid_offset)
            # Add a square at l_pos
            c = (col + 1)/2
            rect = plt.Rectangle(l_pos, 1/scale, 1/scale, color=(c, c, c), fill=True)
            grid.add_patch(rect)





    fig.text(vs + 0.05, 1-0.05, "Player          fc0     fc1   done")
    fig.text(vs + 0.05, 1-0.07, "-----------------------------------------")
    for i, message in enumerate(all_exchange_messages[-18:]):
        fig.text(vs + 0.02, hs-((i+1)*0.03), f"{message}", fontsize=8, wrap=True)
    for i, player in enumerate(step.players):
        color = player_colors[i] if not player.done else "lightgrey"
        fig.text(vs + 0.05, 1-((i+2)*0.05), f"{player}", fontsize=10, wrap=True, family="monospace", color=color)
        p_pos = tuple(p / scale for p in player.pos)
        p_pos = add_tuple(p_pos, player_offsets[i])
        p_pos = add_tuple(p_pos, grid_offset)
        radius = 0.02
        circ = plt.Circle(p_pos, radius=radius, color=color, fill=True)
        if player.done:
            grid.text(*add_tuple(p_pos, (-radius/1.5, -radius/1.5)), f"{i}", fontsize=10, wrap=True, family="monospace")
        grid.add_patch(circ)
    for f, fg in enumerate(step.food_grid):
        for row, frow in enumerate(fg):
            for col, fcount in enumerate(frow):
                if fcount <= 0:
                    continue
                f_pos = (row/scale, col/scale)
                f_pos = add_tuple(f_pos, food_offsets[f])
                f_pos = add_tuple(f_pos, grid_offset)
                circ = plt.Circle(f_pos, radius=0.02*fcount/scale, color=food_colors[f], fill=True)
                grid.add_patch(circ)



with open(args.file, "r") as file:
    lines = file.readlines()
    steps = []
    for i, line in enumerate(lines):
        if "STEP" in line or "game over" in line:
            steps.append(i)
step_slices = []
for i in range(len(steps)-1):
    start = steps[i] + 1
    end = steps[i+1]
    step_slices.append(slice(start, end))

frames = []

num_steps = len(step_slices)
# num_steps = 10  # len(step_slices)
for i in range(num_steps):
    step = Step(i, [], [], [], [])
    food = 0
    grid = step.food_grid
    for line in lines[step_slices[i]]:
        if m := re.match(player_expr, line):
            player, x, y, *ft, done = m.groups()
            p = Player(player, (int(x), int(y)), tuple(float(f) for f in ft), done == "True")
            step.players.append(p)

        if m := re.match(exchange_expr, line):
            giver, amount, food, taker = m.groups()
            step.exchange_messages.append(f"{giver} gave {amount} of {food} to {taker}")
            all_exchange_messages.append(f"{giver} gave {amount} of {food} to {taker}")

        if m := re.match(food_expr, line):
            food = m.groups()
            step.food_grid.append([])
            grid = step.food_grid


        if m := re.match(light_expr, line):
            # light = m.groups()[0]
            step.light_grid.append([])
            grid = step.light_grid
        # Food
        if line.strip().startswith("["):
            grid[-1].append([float(f) for f in line.strip().replace("[", "").replace("]", "").split()])

    frame = plot_step(step)
    frames.append(frame)

# Specify the duration between frames (milliseconds) and save to file:
gif.save(frames, f'{args.file}.gif', duration=200)






#fig = plt.figure()
##axes.append(fig.add_axes([x, y, w, h]))
#vs = 0.6
#hs = 0.6
#axes = [
#    fig.add_axes([0, 0, vs, 1.0]),
#    fig.add_axes([vs, hs, 1-vs, 1-hs]),
#    fig.add_axes([vs, 0, 1-vs, hs]),
#]
#fig.text(0, 0, "hi", fontsize=48, wrap=True)
#for ax in axes:
#    ax.scatter((1, 2), (3, 4))
#plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#
#N = 8
#t = np.linspace(0,2*np.pi, N, endpoint=False)
#r = 0.37
#h = 0.9 - 2*r
#w = h
#X,Y = r*np.cos(t)-w/2.+ 0.5, r*np.sin(t)-h/2.+ 0.5
#
#fig = plt.figure()
#axes = []
#for x,y in zip(X,Y):
#    print(x, y, w, h)
#    axes.append(fig.add_axes([x, y, w, h]))
#
#plt.show()
