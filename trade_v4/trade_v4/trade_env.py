from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
import numpy as np
from random import random, shuffle
from math import floor
from .utils import add_tup, directions, valid_pos, inv_dist, punish_region
from pdb import set_trace as T
from .light import Light
from .spawners import CenterSpawner, FourCornerSpawner, RandomSpawner, FilledCornerSpawner
import sys
from collections import defaultdict


class TradeMetricCollector():
    def __init__(self, env):
        self.rew_health               = 0
        self.rew_nn                   = 0
        self.rew_twonn                = 0
        self.rew_other_survival_bonus = 0
        self.rew_pun                  = 0
        self.rew_mov                  = 0
        self.rew_light                = 0

        self.num_exchanges = [0]*env.food_types
        self.picked_counts = {agent: [0] * env.food_types for agent in env.agents}
        self.placed_counts = {agent: [0] * env.food_types for agent in env.agents}
        self.player_exchanges = {(a, b, f): 0 for a in env.agents for b in env.agents for f in range(env.food_types)}
        self.lifetimes = {agent: 0 for agent in env.agents}

    def collect_lifetimes(self, dones):
        for agent, done in dones.items():
            if not done:
                self.lifetimes[agent] += 1

    def collect_place(self, env, agent, food, actual_place_amount):
        self.placed_counts[agent][food] += actual_place_amount

    def collect_pick(self, env, agent, x, y, food, agent_id):
        exchange_amount = env.compute_exchange_amount(x, y, food, agent_id)
        if exchange_amount > 0:
            for i, other_agent in enumerate(env.agents):
                self.player_exchanges[(other_agent, agent, food)] += env.table[x, y, food, i]
        self.num_exchanges[food] += exchange_amount
        self.picked_counts[agent][food] += env.compute_pick_amount(x, y, food, agent_id)
        pass

    def collect_rew(self, env, health, nn, twonn, other_survival_bonus, pun_rew, mov_rew, light):
        self.rew_health               += health
        self.rew_nn                   += nn
        self.rew_twonn                += twonn
        self.rew_other_survival_bonus += other_survival_bonus
        self.rew_pun                  += pun_rew
        self.rew_mov                  += mov_rew
        self.rew_light                += light

METABOLISM=0.1
PLACE_AMOUNT = 0.5
SCALE_DOWN = 30

NUM_ITERS = 100
ndir = len(directions)

class Trade(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        print(f"Creating Trade environment {env_config}")
        gx, gy = self.grid_size    = env_config.get("grid", (7, 7))
        self.food_types            = env_config.get("food_types", 2)
        num_agents                 = env_config.get("num_agents", 2)
        self.max_steps             = env_config.get("episode_length", 100)
        self.vocab_size            = env_config.get("vocab_size", 0)
        self.window_size           = env_config.get("window", (3, 3))
        self.dist_coeff            = env_config.get("dist_coeff", 0.0)
        self.move_coeff            = env_config.get("move_coeff", 0.0)
        self.death_prob            = env_config.get("death_prob", 0.1)
        self.day_night_cycle       = env_config.get("day_night_cycle", False)
        self.day_steps             = env_config.get("day_steps", 20)
        self.night_time_death_prob = env_config.get("night_time_death_prob", 0.1)
        self.punish                = env_config.get("punish", False)
        self.punish_coeff          = env_config.get("punish_coeff", 3)
        self.survival_bonus        = env_config.get("survival_bonus", 0.0)
        self.respawn               = env_config.get("respawn", False)
        self.spawn_agents          = env_config.get("spawn_agents", "center")
        self.twonn_coeff           = env_config.get("twonn_coeff", 0.0)
        self.light_coeff           = env_config.get("light_coeff", 1.0)
        self.health_baseline       = env_config.get("health_baseline", False)
        self.policy_mapping_fn     = env_config.get("policy_mapping_fn")
        self.food_env_spawn        = env_config.get("food_env_spawn")
        self.food_agent_start      = env_config.get("food_agent_start", 1)
        self.padded_grid_size      = add_tup(self.grid_size, add_tup(self.window_size, self.window_size))
        self.light                 = Light(self.grid_size, 2/self.day_steps)
        super().__init__()


        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_group_mapping = defaultdict(list)
        for agent in self.agents:
            self.agent_group_mapping[self.policy_mapping_fn(agent)].append(agent)
        self.policies = sorted(list(self.agent_group_mapping.keys()))
        # (self + policies) * (food frames and pos frame)
        food_frame_and_agent_channels = (len(self.agent_group_mapping.keys())+1) * (self.food_types+1)
        # x, y + agents_and_foods + food frames + comms
        self.channels = 2 + food_frame_and_agent_channels + (self.food_types) + (self.vocab_size) + int(self.punish) + int(self.day_night_cycle)
        self.agent_food_counts = dict()
        self.MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
        if self.punish:
            self.MOVES.append("PUNISH")
        for f in range(self.food_types):
            self.MOVES.extend([f"PICK_{f}", f"PLACE_{f}"])
        self.MOVES.extend([f"COMM_{c}" for c in range(self.vocab_size)])
        self.MOVES.append("NONE")
        self.num_actions = len(self.MOVES)
        self.communications = {}

        self.agent_spawner = {
            "center": CenterSpawner(self.grid_size),
            "corner": FourCornerSpawner(self.grid_size),
            "random": RandomSpawner(self.grid_size)
        }[self.spawn_agents]

        self.food_spawner = FilledCornerSpawner(self.grid_size)



        self.action_space = Discrete(self.num_actions)
        self.obs_size = (*add_tup(add_tup(self.window_size, self.window_size), (1, 1)), self.channels)
        self.observation_space = Box(low=np.full(self.obs_size, -1, dtype=np.float32), high=np.full(self.obs_size, 10))
        self._skip_env_checking = True

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def spawn_food(self):
        fc = self.food_env_spawn if self.respawn else 10
        food_counts = [(0, fc), (0, fc), (1, fc), (1, fc)]
        num_piles = 100

        for i in range(num_piles):
            spawn_spots = self.food_spawner.gen_poses()
            for spawn_spot, (ft, fc) in zip(spawn_spots, food_counts):
                fx, fy = spawn_spot
                self.table[fx, fy, ft, len(self.agents)] += fc / num_piles

    def reset(self):
        self.light.reset()
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.moved_last_turn = {agent: False for agent in self.agents}
        gx, gy = self.grid_size
        # use the last slot in the agents dimension to specify naturally spawning food
        self.table = np.zeros((*self.grid_size, self.food_types, len(self.agents)+1), dtype=np.float32)

        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        self.spawn_food()
        self.agent_spawner.reset()
        spawn_spots = self.agent_spawner.gen_poses()
        self.agent_positions = {agent: spawn_spot for agent, spawn_spot in zip(self.agents, spawn_spots)}
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.agent_food_counts = {agent: [self.food_agent_start for f in range(self.food_types)] for agent in self.agents}
        self.mc = TradeMetricCollector(self)
        return {agent: self.compute_observation(agent) for agent in self.agents}

    def render(self, mode="human", out=sys.stdout):
        for agent in self.agents:
            out.write(f"{agent}: {self.agent_positions[agent]} {[round(fc, 2) for fc in self.agent_food_counts[agent]]} {self.compute_done(agent)}\n")
        for food in range(self.food_types):
            out.write(f"food{food}:\n {self.table[:,:,food].sum(axis=2).round(2)}\n")
        out.write(f"Total exchanged so far: {self.mc.num_exchanges}\n")
        if self.day_night_cycle:
            out.write(f"Light:\n {self.light.frame().round(2)}\n")
        for agent, comm in self.communications.items():
            if comm and max(comm) >= 1:
                out.write(f"{agent} said {comm.index(1)}\n")

    def compute_observation(self, agent=None):
        ax, ay = self.agent_positions[agent]
        wx, wy = self.window_size
        gx, gy = self.grid_size

        minx, maxx = ax, ax+(2*wx)+1
        miny, maxy = ay, ay+(2*wy)+1
        food_frames = self.table.sum(axis=3).transpose(2, 0, 1)  # frame for each food
        comm_frames = np.zeros((self.vocab_size, *self.grid_size), dtype=np.float32)
        self_pos_frame   = np.zeros(self.grid_size, dtype=np.float32)
        self_pos_frame[ax, ay] = 1
        self_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        pol_pos_frames   = {p: np.zeros(self.grid_size, dtype=np.float32) for p in self.policies}
        pol_food_frames  = {p: np.zeros((self.food_types, *self.grid_size), dtype=np.float32) for p in self.policies}
        for i, a in enumerate(self.agents):
            if self.compute_done(a):
                continue
            oax, oay = self.agent_positions[a]
            comm_frames[:, oax, oay] = self.communications[a]

            if a != agent:
                pol = self.policy_mapping_fn(a)
                pol_pos_frames[pol][oax, oay] += 1
                pol_food_frames[pol][:, oax, oay] += self.agent_food_counts[a]
            else:
                self_food_frames[:,  oax, oay] += self.agent_food_counts[a]

        pol_frames = np.concatenate([np.stack([*pol_food_frames[p], pol_pos_frames[p]]) for p in self.policies])
        agent_and_food_frames = np.stack([*pol_frames, self_pos_frame, *self_food_frames])

        if self.punish:
            pun_frames = np.sum(self.punish_frames, axis=0)[None, :, :]
        else:
            pun_frames = np.zeros((0, *self.grid_size), dtype=np.float32)

        if self.day_night_cycle:
            light_frames = self.light.frame()[None, :, :] * SCALE_DOWN
        else:
            light_frames = np.zeros((0, *self.grid_size), dtype=np.float32)


        xpos_frame = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gx
        ypos_frame = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gy

        frames = np.stack([*food_frames, *agent_and_food_frames, xpos_frame, ypos_frame, *light_frames, *pun_frames, *comm_frames], axis=2)
        padded_frames = np.full((*self.padded_grid_size, frames.shape[2]), -1, dtype=np.float32)
        padded_frames[wx:(gx+wx), wy:(gy+wy), :] = frames
        obs = padded_frames[minx:maxx, miny:maxy, :] / SCALE_DOWN
        return obs

    def compute_done(self, agent):
        if self.dones[agent] or self.steps >= self.max_steps:
            return True
        return False


    def compute_reward(self, agent):
        # reward for each living player
        rew = 0
        if self.compute_done(agent):
            return rew
        pos = self.agent_positions[agent]
        # same_poses = 0
        dists = [0, 0]
        other_survival_bonus = 0
        punishment = 0
        for aid, a in enumerate(self.agents):
            if not self.compute_done(a) and a != agent:
                dists.append(inv_dist(pos, self.agent_positions[a]))
                other_survival_bonus += self.survival_bonus
                punishment += self.punish_frames[aid, pos[0], pos[1]]
        dists.sort()

        if self.health_baseline:
            #num_of_food_types = sum(1 for f in self.agent_food_counts[agent] if f >= 0.1)
            #health = [0, 1, 10][num_of_food_types]
            health = 1 if min(self.agent_food_counts[agent]) >= 0.1 else 0.5
        else:
            health = 1

        light_rew = 0 if self.light.contains(self.agent_positions[agent]) else -self.light_coeff

        nn_rew    =  (self.dist_coeff * dists[-1])
        twonn_rew = -(self.twonn_coeff * dists[-2])
        pun_rew   = -self.punish_coeff * punishment
        mov_rew   = -self.move_coeff * int(self.moved_last_turn[agent])

        # Remember to update this function whenever you add a new reward
        self.mc.collect_rew(self, health, nn_rew, twonn_rew, other_survival_bonus, pun_rew, mov_rew, light_rew)

        rew  = health + nn_rew + twonn_rew + other_survival_bonus + pun_rew + mov_rew + light_rew
        return rew

    def compute_exchange_amount(self, x: int, y: int, food: int, picker: int):
        return sum(count for a, count in enumerate(self.table[x][y][food]) if a != picker and a != len(self.agents))

    def compute_pick_amount(self, x: int, y: int, food: int, picker: int):
        return self.table[x][y][food][len(self.agents)]

    def update_dones(self):
        for agent in self.agents:
            if self.compute_done(agent):
                continue
            if max(self.agent_food_counts[agent]) < 0.1:
                self.dones[agent] = True
            else:
                for f in self.agent_food_counts[agent]:
                    if f < 0.1 and random() < self.death_prob:
                        self.dones[agent] = True
            #if not self.light.contains(self.agent_positions[agent]) and random() < self.night_time_death_prob:
            #    self.dones[agent] = True

    def step(self, actions):
        self.light.step_light()
        # all agents execute their action
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        random_order = list(actions.keys())
        shuffle(random_order)
        # placed goods will not be available until next turn
        place_table = np.zeros(self.table.shape, dtype=np.float32)
        gx, gy = self.grid_size
        for agent in random_order:
            action = actions[agent]
            # MOVEMENT
            self.moved_last_turn[agent] = False
            x, y = self.agent_positions[agent]
            aid: int = self.agents.index(agent)
            if action in range(0, ndir):
                new_pos = add_tup(self.agent_positions[agent], directions[action])
                if valid_pos(new_pos, self.grid_size):
                    self.agent_positions[agent] = new_pos
                    self.moved_last_turn[agent] = True
            # punish
            elif action in range(ndir, ndir + int(self.punish)):
                x_pun_region, y_pun_region = punish_region(x, y, *self.grid_size)
                self.punish_frames[aid, x_pun_region, y_pun_region] = 1
            elif action in range(ndir + int(self.punish), ndir + int(self.punish) + (self.food_types * 2)):
                pick = ((action - ndir - int(self.punish)) % 2 == 0)
                food = floor((action - ndir - int(self.punish)) / 2)
                if pick:
                    self.mc.collect_pick(self, agent, x, y, food, aid)
                    self.agent_food_counts[agent][food] += np.sum(self.table[x, y, food])
                    self.table[x, y, food, :] = 0
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    actual_place_amount = PLACE_AMOUNT
                    self.agent_food_counts[agent][food] -= actual_place_amount
                    place_table[x, y, food, aid] += actual_place_amount
                    self.mc.collect_place(self, agent, food, actual_place_amount)
            # last action is noop
            elif action in range(4 + self.food_types * 2 + int(self.punish), self.num_actions-1):
                symbol = action - (self.food_types * 2) - 4
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1
            self.agent_food_counts[agent] = [max(x - METABOLISM, 0) for x in self.agent_food_counts[agent]]

        self.update_dones()


        # Once agents complete all actions, add placed food to table
        self.table = self.table + place_table
        self.steps += 1
        if self.respawn and self.light.dawn():
            self.spawn_food()

        obs = {agent: self.compute_observation(agent) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        self.mc.collect_lifetimes(dones)
        rewards = {agent: self.compute_reward(agent) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        infos = {}
        return obs, rewards, dones, infos
