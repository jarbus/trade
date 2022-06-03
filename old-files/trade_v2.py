from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
import numpy as np
from random import randint
from math import floor
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import add_tup, directions, valid_pos, inv_dist
from pdb import set_trace as T

NUM_ITERS = 100
PLACE_AMOUNT = 0.5
ndir = len(directions)

class Trade(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        self.food_types = env_config.get("food_types", 2)
        num_agents = env_config.get("num_agents", 2)
        self.max_steps = env_config.get("episode_length", 100)
        self.vocab_size = env_config.get("vocab_size", 0)
        self.grid_size = env_config.get("grid", (1, 5))
        self.window_size = env_config.get("window", (1, 5))
        self.padded_grid_size = add_tup(self.grid_size, add_tup(self.window_size, self.window_size))
        super().__init__()
        self.channels = 4 + (self.food_types * 3) + (self.vocab_size*num_agents)
        self.agent_food_counts = dict()
        self.MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
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
        # TODO better starting pos
        gx, gy = self.grid_size
        self.agent_positions = {agent: (randint(0, gx-1), randint(0, gy-1)) for agent in self.agents}
        # Grid of placed food
        self.table = np.zeros((*self.grid_size, self.food_types, len(self.agents)), dtype=np.float32)
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.num_exchanges = [0]*self.food_types
        self.lifetimes = {agent: 0 for agent in self.agents}
        self.agent_food_counts = dict()
        for i, agent in enumerate(self.agents):
            self.agent_food_counts[agent] = []
            for j in range(self.food_types):
                if i == j:
                    self.agent_food_counts[agent].append(10)
                else:
                    self.agent_food_counts[agent].append(1)
        return {agent: self.compute_observation(agent) for agent in self.agents}

    def render(self, mode="human"):
        for agent in self.agents:
            print(f"{agent}: {self.agent_positions[agent]} {self.agent_food_counts[agent]} {self.compute_done(agent)}")
        for food in range(self.food_types):
            print(f"food{food}: {self.table[:,:,food].sum(axis=2)}")
        print(f"Total exchanged so far: {self.num_exchanges}")
        for agent, comm in self.communications.items():
            if comm and max(comm) >= 1:
                print(f"{agent} said {comm.index(1)}")

    def compute_observation(self, agent=None):
        # stores action of current agent
        ax, ay = self.agent_positions[agent]
        wx, wy = self.window_size
        gx, gy = self.grid_size

        # TODO ADD COMMUNICATION
        minx, maxx = ax, ax+(2*wx)+1
        miny, maxy = ay, ay+(2*wy)+1
        food_frames = self.table.sum(axis=3).transpose(2, 0, 1)  # frame for each food
        other_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        self_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        comm_frames = np.zeros((self.vocab_size, *self.grid_size), dtype=np.float32)
        agent_frame = np.zeros(self.grid_size, dtype=np.float32)
        self_frame = np.zeros(self.grid_size, dtype=np.float32)
        self_frame[ax, ay] = 1
        for a in self.agents:
            oax, oay = self.agent_positions[a]
            comm_frames[:, oax, oay] = self.communications[a]
            if a != agent:
                agent_frame[oax, oay] += 1
            for f in range(self.food_types):
                if a != agent:
                    other_food_frames[f, oax, oay] += self.agent_food_counts[a][f]
                else:
                    self_food_frames[f, oax, oay] += self.agent_food_counts[a][f]
        xpos_frame = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gx
        ypos_frame = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gy
        frames = np.stack([*food_frames, *self_food_frames, *other_food_frames, *comm_frames, agent_frame, self_frame, xpos_frame, ypos_frame])
        padded_frames = np.full((frames.shape[0], *self.padded_grid_size), -1, dtype=np.float32)
        padded_frames[:, wx:(gx+wx), wy:(gy+wy)] = frames
        obs = padded_frames[:, minx:maxx, miny:maxy] / 30
        return obs

    def compute_done(self, agent):
        if sum(f < 0.1 for f in self.agent_food_counts[agent]) >= 2 or self.steps >= self.max_steps:
            return True
        return False

    def compute_reward(self, agent):
        # reward for each living player
        rew = 0
        if self.compute_done(agent):
            return rew
        pos = self.agent_positions[agent]
        for a in self.agents:
            if not self.compute_done(a):
                rew += inv_dist(pos, self.agent_positions[a])
        return rew

    def compute_exchange_amount(self, x: int, y: int, food: int, picker: int):
        return sum(count for a, count in enumerate(self.table[x][y][food]) if a != picker)

    def step(self, actions):
        # all agents execute their action
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        for (agent, action) in actions.items():
            # MOVEMENT
            if action in range(0, ndir):
                new_pos = add_tup(self.agent_positions[agent], directions[action])
                if valid_pos(new_pos, self.grid_size):
                    self.agent_positions[agent] = new_pos
            elif action in range(ndir, ndir + self.food_types * 2):
                pick = ((action - ndir) % 2 == 0)
                food = floor((action - ndir) / 2)
                x, y = self.agent_positions[agent]
                aid: int = self.agents.index(agent)
                if pick:
                    self.num_exchanges[food] += self.compute_exchange_amount(x, y, food, aid)
                    self.agent_food_counts[agent][food] += np.sum(self.table[x, y, food])
                    self.table[x, y, food, :] = 0
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    self.agent_food_counts[agent][food] -= PLACE_AMOUNT
                    self.table[x, y, food, aid] += PLACE_AMOUNT
            # last action is noop
            elif action in range(4 + self.food_types * 2, self.num_actions-1):
                symbol = action - (self.food_types * 2) - 4
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1

        for agent in self.agents:
            self.agent_food_counts[agent] = [x - 0.1 for x in self.agent_food_counts[agent]]

        self.steps += 1

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
        for a in env.agents:
            if env.compute_done(a):
                self.action_counts[env.MOVES[episode.last_action_for(a)]] += 1



    def on_episode_end(self, worker, base_env, policies, episode, **kwargs,):
        env = base_env.get_unwrapped()[0]
        episode.custom_metrics["grid_size"] = env.grid_size
        for food, count in enumerate(env.num_exchanges):
            episode.custom_metrics[f"exchange_{food}"] = count
        for symbol, count in enumerate(self.comm_history):
            episode.custom_metrics[f"comm_{symbol}"] = count
        for agent, lifetime in env.lifetimes.items():
            episode.custom_metrics[f"{agent}_lifetime"] = lifetime
        episode.custom_metrics[f"avg_avg_dist"] = sum(self.agent_dists) / len(self.agent_dists)
        total_number_of_actions = sum(self.action_counts.values())
        if total_number_of_actions > 0:
            for act, count in self.action_counts.items():
                episode.custom_metrics[f"percent_{act}"] = count / total_number_of_actions



if __name__ == "__main__":
    num_agents = 3
    env_config = {"window": (2, 2),
                  "grid": (2, 3),
                  "food_types": num_agents,
                  "num_agents": num_agents,
                  "episode_length": 100,
                  "vocab_size": 0}
    tenv = Trade(env_config)
    obs = tenv.reset()
    for i in range(100):
        actions = {}
        for agent in tenv.agents:
            tenv.render()
            for i, m in enumerate(tenv.MOVES):
                print(f"{i, m}", end="")
            print(f"\nAction for {agent}: ", end="")
            action = int(input())
            actions[agent] = action
        obss, rews, dones, infos = tenv.step(actions)
        print(obss["player_0"])
        if dones["__all__"]:
            print("game over")
            break
