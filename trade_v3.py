from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
import numpy as np
from random import randint, random, shuffle, choice
from math import floor
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import add_tup, directions, valid_pos, inv_dist, two_combos, punish_region
from pdb import set_trace as T
import sys
from time import time

NUM_ITERS = 100
PLACE_AMOUNT = 0.5
ndir = len(directions)

class Trade(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        print(f"Creating Trade environment {env_config}")
        self.food_types = env_config.get("food_types", 2)
        num_agents = env_config.get("num_agents", 2)
        self.max_steps = env_config.get("episode_length", 100)
        self.vocab_size = env_config.get("vocab_size", 0)
        self.grid_size = env_config.get("grid", (1, 5))
        self.window_size = env_config.get("window", (1, 5))
        self.dist_coeff = env_config.get("dist_coeff", 0.5)
        self.move_coeff = env_config.get("move_coeff", 0.5)
        self.death_prob = env_config.get("death_prob", 0.1)
        self.punish = env_config.get("punish", True)
        self.punish_coeff = env_config.get("punish_coeff", 3)
        self.survival_bonus = env_config.get("survival_bonus", 0.0)
        self.respawn = env_config.get("respawn", True)
        self.random_start = env_config.get("random_start", True)
        self.padded_grid_size = add_tup(self.grid_size, add_tup(self.window_size, self.window_size))
        super().__init__()
        # x, y + self, other + selffc, otherfc + food frames + comms
        self.channels = 2 + 2 + (2*self.food_types) + (self.food_types) + (self.vocab_size) + int(self.punish)
        self.agent_food_counts = dict()
        self.MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
        if self.punish:
            self.MOVES.append("PUNISH")
        for f in range(self.food_types):
            self.MOVES.extend([f"PICK_{f}", f"PLACE_{f}"])
        self.MOVES.extend([f"COMM_{c}" for c in range(self.vocab_size)])
        self.MOVES.append("NONE")
        self.num_actions = len(self.MOVES)
        self.num_exchanges = []
        self.communications = {}

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.action_space = Discrete(self.num_actions)
        self.obs_size = (self.channels, *add_tup(add_tup(self.window_size, self.window_size), (1, 1)))
        self.observation_space = Box(low=np.full(self.obs_size, -1, dtype=np.float32), high=np.full(self.obs_size, 10))
        self._skip_env_checking = True

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.moved_last_turn = {agent: False for agent in self.agents}
        self.picked_counts = {agent: [0] * self.food_types for agent in self.agents}
        self.placed_counts = {agent: [0] * self.food_types for agent in self.agents}
        gx, gy = self.grid_size
        # use the last slot in the agents dimension to specify naturally spawning food
        self.table = np.zeros((*self.grid_size, self.food_types, len(self.agents)+1), dtype=np.float32)

        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        # Usage: random.choice(self.spawn_spots[0-4])
        self.spawn_spots = [[(0,1), (0, 1)], [(0,1), (gy-2,gy-1)], [(gx-2,gx-1), (0,1)], [(gx-2,gx-1), (gy-2,gy-1)]]
        self.spawn_spots = [two_combos(xs, ys) for (xs, ys) in self.spawn_spots]
        fc = 4 if self.respawn else 10
        food_counts = [(0, fc), (0, fc), (1, fc), (1, fc)]
        if self.random_start:
            shuffle(food_counts)
            shuffle(self.spawn_spots)
        for spawn_spot, (ft, fc) in zip(self.spawn_spots, food_counts):
            fx, fy = choice(spawn_spot)
            self.table[fx, fy, ft, len(self.agents)] = fc
        self.agent_positions = {agent: choice(spawn_spots) for agent, spawn_spots in zip(self.agents, self.spawn_spots)}
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.num_exchanges = [0]*self.food_types
        self.player_exchanges = {(a, b, f): 0 for a in self.agents for b in self.agents for f in range(self.food_types)}
        self.lifetimes = {agent: 0 for agent in self.agents}
        self.agent_food_counts = {"player_0": [1, 1], "player_1": [1, 1], "player_2": [1, 1], "player_3": [1, 1]}
        return {agent: self.compute_observation(agent) for agent in self.agents}

    def render(self, mode="human", out=sys.stdout):
        for agent in self.agents:
            out.write(f"{agent}: {self.agent_positions[agent]} {[round(fc, 2) for fc in self.agent_food_counts[agent]]} {self.compute_done(agent)}\n")
        for food in range(self.food_types):
            out.write(f"food{food}:\n {self.table[:,:,food].sum(axis=2)}\n")
        out.write(f"Total exchanged so far: {self.num_exchanges}\n")
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
        other_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        self_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        #agent_food_frames = np.zeros((self.food_types*len(self.agents), *self.grid_size), dtype=np.float32)
        comm_frames = np.zeros((self.vocab_size, *self.grid_size), dtype=np.float32)
        #agent_frames = np.zeros((len(self.agents), *self.grid_size), dtype=np.float32)
        self_frame = np.zeros(self.grid_size, dtype=np.float32)
        other_frame = np.zeros(self.grid_size, dtype=np.float32)
        self_frame[ax, ay] = 1
        for i, a in enumerate(self.agents):
            if self.compute_done(a):
                continue
            oax, oay = self.agent_positions[a]
            comm_frames[:, oax, oay] = self.communications[a]
            if a != agent:
                other_frame[oax, oay] += 1
                for f in range(self.food_types):
                    other_food_frames[f, oax, oay] += self.agent_food_counts[a][f]
            else:
                for f in range(self.food_types):
                    self_food_frames[f, oax, oay] += self.agent_food_counts[a][f]
        xpos_frame = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gx
        ypos_frame = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gy

        if self.punish:
            pun_frames = np.sum(self.punish_frames, axis=0)[None, :, :]
        else:
            pun_frames = np.zeros((0, *self.grid_size), dtype=np.float32)

        frames = np.stack([*food_frames, *self_food_frames, *pun_frames,  *other_food_frames, *comm_frames, other_frame, self_frame, xpos_frame, ypos_frame])
        padded_frames = np.full((frames.shape[0], *self.padded_grid_size), -1, dtype=np.float32)
        padded_frames[:, wx:(gx+wx), wy:(gy+wy)] = frames
        obs = padded_frames[:, minx:maxx, miny:maxy] / 30
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
                #rew += self.dist_coeff * inv_dist(pos, self.agent_positions[a])
                dists.append(inv_dist(pos, self.agent_positions[a]))
                other_survival_bonus += self.survival_bonus
                punishment += self.punish_frames[aid, pos[0], pos[1]]
                #if self.agent_positions[a] == pos:
                #    same_poses += 1
        dists.sort()
        rew = 1 + (self.dist_coeff * dists[-1]) + other_survival_bonus
        rew -= self.punish_coeff * punishment
        rew -= self.move_coeff * int(self.moved_last_turn[agent])
        #if same_poses > 2:
        #    rew -= same_poses
        return rew

    def compute_exchange_amount(self, x: int, y: int, food: int, picker: int):
        return sum(count for a, count in enumerate(self.table[x][y][food]) if a != picker and a != len(self.agents))

    def compute_pick_amount(self, x: int, y: int, food: int, picker: int):
        return self.table[x][y][food][len(self.agents)]

    def step(self, actions):
        # all agents execute their action
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        random_order = list(actions.keys())
        shuffle(random_order)
        # placed goods will not be available until next turn
        #place_table = np.zeros(self.table.shape, dtype=np.float32)
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
                    exchange_amount = self.compute_exchange_amount(x, y, food, aid)
                    if exchange_amount > 0:
                        for i, other_agent in enumerate(self.agents):
                            self.player_exchanges[(other_agent, agent, food)] += self.table[x, y, food, i]
                    self.num_exchanges[food] += exchange_amount
                    self.picked_counts[agent][food] += self.compute_pick_amount(x, y, food, aid)
                    self.agent_food_counts[agent][food] += np.sum(self.table[x, y, food])
                    self.table[x, y, food, :] = 0
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    actual_place_amount = PLACE_AMOUNT
                    self.agent_food_counts[agent][food] -= actual_place_amount
                    self.table[x, y, food, aid] += actual_place_amount
                    self.placed_counts[agent][food] += actual_place_amount
            # last action is noop
            elif action in range(4 + self.food_types * 2 + int(self.punish), self.num_actions-1):
                symbol = action - (self.food_types * 2) - 4
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1

        for agent in self.agents:
            if self.compute_done(agent):
                continue
            self.agent_food_counts[agent] = [max(x - 0.1, 0) for x in self.agent_food_counts[agent]]
            if max(self.agent_food_counts[agent]) < 0.1:
                self.dones[agent] = True
            else:
                for f in self.agent_food_counts[agent]:
                    if f < 0.1 and random() < self.death_prob:
                        self.dones[agent] = True

        # Once agents complete all actions, add placed food to table
        #self.table = self.table + place_table
        # RESET FOOD EVERY TEN ITERS
        self.steps += 1
        if self.respawn and self.steps % 15 == 0:
            fc = 4 if self.respawn else 10
            food_counts = [(0, fc), (0, fc), (1, fc), (1, fc)]
            for spawn_spot, (ft, fc) in zip(self.spawn_spots, food_counts):
                fx, fy = choice(spawn_spot)
                self.table[fx, fy, ft, len(self.agents)] = fc
        #if self.steps == 30:
        #    self.table[0, 0, 0, len(self.agents)] = 10
        #    self.table[4, 4, 1, len(self.agents)] = 10

        obs = {agent: self.compute_observation(agent) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        for agent, done in dones.items():
            if not done:
                self.lifetimes[agent] += 1
        rewards = {agent: self.compute_reward(agent) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        infos = {}
        return obs, rewards, dones, infos

class TradeCallback(DefaultCallbacks):

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0]
        self.comm_history = [0 for i in range(env.vocab_size)]
        self.agent_dists = []
        self.action_counts = {act: 0 for act in env.MOVES}
        self.punish_counts = {agent: 0 for agent in env.agents}

    def on_episode_step(self, worker, base_env, policies, episode, **kwargs):
        # there is a bug where on_episode_step gets called where it shouldn't
        env = base_env.get_unwrapped()[0]
        for agent, comm in env.communications.items():
            if comm and max(comm) == 1:
                symbol = comm.index(1)
                self.comm_history[symbol] += 1
        dists = {}
        for i, a in enumerate(env.agents[:-1]):
            for j, b in enumerate(env.agents[i+1:]):
                if env.compute_done(a) or env.compute_done(b):
                    continue
                dists[(a,b)] = inv_dist(env.agent_positions[a], env.agent_positions[b])
        self.agent_dists.append(sum(dists.values()) / max(1, len(dists.values())))
        self.agent_grid = np.zeros(env.grid_size)
        for a in env.agents:
            if not env.compute_done(a):
                self.action_counts[env.MOVES[episode.last_action_for(a)]] += 1
                self.agent_grid[env.agent_positions[a]] += 1
        for aid, a in enumerate(env.agents):
            ax, ay = env.agent_positions[a]
            self.punish_counts[a] += np.sum(self.agent_grid * env.punish_frames[aid])\
                - env.punish_frames[aid, ax, ay]  # remove self-punish




    def on_episode_end(self, worker, base_env, policies, episode, **kwargs,):
        env = base_env.get_unwrapped()[0]
        if not all(env.dones.values()):
            return
        episode.custom_metrics["grid_size"] = env.grid_size
        for food, count in enumerate(env.num_exchanges):
            episode.custom_metrics[f"exchange_{food}"] = count
        for symbol, count in enumerate(self.comm_history):
            episode.custom_metrics[f"comm_{symbol}"] = count
        for agent in env.agents:
            episode.custom_metrics[f"{agent}_punishes"] = self.punish_counts[agent]
            episode.custom_metrics[f"{agent}_lifetime"] = env.lifetimes[agent]
            episode.custom_metrics[f"{agent}_food_imbalance"] = \
                max(env.agent_food_counts[agent]) / max(1, min(env.agent_food_counts[agent]))
            total_agent_exchange = {"give": 0, "take": 0}
            for other_agent in env.agents:
                other_agent_exchange = {"give": 0, "take": 0}
                for food in range(env.food_types):
                    give = env.player_exchanges[(agent, other_agent, food)]
                    take = env.player_exchanges[(other_agent, agent, food)]
                    other_agent_exchange["give"] += give
                    other_agent_exchange["take"] += take
                    if other_agent != agent:
                        total_agent_exchange["give"] += give
                        total_agent_exchange["take"] += take
                #episode.custom_metrics[f"{agent}_take_from_{other_agent}"] = other_agent_exchange["take"]
                #episode.custom_metrics[f"{agent}_give_to_{other_agent}"] = other_agent_exchange["give"]
                episode.custom_metrics[f"{agent}_mut_exchange_{other_agent}"] =\
                   min(other_agent_exchange["take"], other_agent_exchange["give"])
            #episode.custom_metrics[f"{agent}_take_from_all"] = total_agent_exchange["take"]
            #episode.custom_metrics[f"{agent}_give_to_all"] = total_agent_exchange["give"]
            episode.custom_metrics[f"{agent}_mut_exchange_total"] =\
                min(total_agent_exchange["take"], total_agent_exchange["give"])
            for food in range(env.food_types):
                episode.custom_metrics[f"{agent}_PICK_{food}"] = env.picked_counts[agent][food]
                episode.custom_metrics[f"{agent}_PLACE_{food}"] = env.placed_counts[agent][food]

        episode.custom_metrics[f"avg_avg_dist"] = sum(self.agent_dists) / len(self.agent_dists)
        total_number_of_actions = sum(self.action_counts.values())
        if total_number_of_actions > 0:
            for act, count in self.action_counts.items():
                episode.custom_metrics[f"percent_{act}"] = count / total_number_of_actions



if __name__ == "__main__":
    num_agents = 4
    env_config = {"window": (2, 2),
                  "grid": (5, 5),
                  "food_types": 2,
                  "num_agents": num_agents,
                  "episode_length": 200,
                  "vocab_size": 0}
    tenv = Trade(env_config)
    obs = tenv.reset()
    start = time()
    for i in range(100):
        actions = {}
        for agent in tenv.agents:
            tenv.render()
            for i, m in enumerate(tenv.MOVES):
                print(f"{i, m}", end="")
            print(f"\nAction for {agent}: ", end="")
            #action = int(input())
            action = 4
            actions[agent] = action
        obss, rews, dones, infos = tenv.step(actions)
        # print("player_0 obs")
        # print(obss["player_0"])
        # print("player_1 obs")
        # print(obss["player_1"])
        if dones["__all__"]:
            print("game over")

            print(f"time: {time() - start}")
            break
