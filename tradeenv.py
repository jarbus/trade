from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
import numpy as np
from math import floor
from collections import defaultdict
from ray.rllib.agents.callbacks import DefaultCallbacks

NUM_ITERS = 100
PLACE_AMOUNT = 0.5

class Trade(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        self.food_types = env_config.get("food_types", 2)
        num_agents = env_config.get("num_agents", 2)
        self.max_steps = env_config.get("episode_length", 100)
        self.vocab_size = env_config.get("vocab_size", 6)
        super().__init__()
        self.num_inputs = self.food_types * (num_agents+1) + (self.vocab_size*num_agents)
        self.agent_food_counts = dict()
        self.MOVES = []
        for f in range(self.food_types):
            self.MOVES.extend([f"PICK_{f}", f"PLACE_{f}"])
        self.MOVES.extend([f"COMM_{c}" for c in range(self.vocab_size)])
        self.MOVES.append("NONE")
        self.num_actions = len(self.MOVES)
        self.num_exchanges = []
        self.exchange_table = []
        self.communications = {}

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(low=np.array([-1 for _ in range(self.num_inputs)]), high=np.array([10 for _ in range(self.num_inputs)]))
        self._skip_env_checking = True

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.table = [0]*self.food_types
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.num_exchanges = [0]*self.food_types
        self.exchange_table = [defaultdict(int) for f in range(self.food_types)]
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
            print(f"{agent}: {self.agent_food_counts[agent]} {self.compute_done(agent)}")
        print(f"table: {self.table}")
        print(f"Total exchanged so far: {self.num_exchanges}")
        for agent, comm in self.communications:
            if max(comm) >= 1:
                print(f"{agent} said {comm.index(1)}")

    def compute_observation(self, agent=None):
        # stores action of current agent
        curr_state = self.table.copy()
        if agent:
            curr_state.extend([*self.agent_food_counts[agent]])
            curr_state.extend([*self.communications[agent]])
        for other_agent in self.agents:
            if agent != other_agent:
                curr_state.extend(self.agent_food_counts[other_agent])
                curr_state.extend([*self.communications[other_agent]])
        curr_state = [s / 10 for s in curr_state]
        return curr_state

    def compute_done(self, agent):
        if min(self.agent_food_counts[agent]) <= 0.1 or self.steps >= self.max_steps:
            return True
        return False

    def compute_reward(self, agent):
        if self.compute_done(agent):
            return 0
        return 1

    def compute_exchange_amount(self, food: int, picker:int):
        return sum(self.exchange_table[food][agent] for agent in self.exchange_table[food].keys() if agent != picker)

    def step(self, actions):
        # all agents execute their action
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        for (agent, action) in actions.items():
            if action in range(self.food_types * 2):
                pick = (action % 2 == 0)
                food = floor(action / 2)
                if pick:
                    self.agent_food_counts[agent][food] += self.table[food]
                    self.table[food] = 0
                    self.num_exchanges[food] += self.compute_exchange_amount(food, agent)
                    self.exchange_table[food] = defaultdict(int)
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    self.agent_food_counts[agent][food] -= PLACE_AMOUNT
                    self.table[food] += PLACE_AMOUNT
                    self.exchange_table[food][agent] += PLACE_AMOUNT
            elif action in range(self.food_types * 2, self.num_actions-1): # last action is noop
                symbol = action - self.food_types * 2
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1

        for agent in self.agents:
            self.agent_food_counts[agent] = [x - 0.1 for x in self.agent_food_counts[agent]]

        self.steps += 1

        obs = {agent: self.compute_observation(agent) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        rewards = {agent: self.compute_reward(agent) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        infos = {}
        return obs, rewards, dones, infos

class TradeCallback(DefaultCallbacks):

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        self.comm_history = [0 for i in range(base_env.get_unwrapped()[0].vocab_size)]

    def on_episode_step(self, worker, base_env, policies, episode, **kwargs):
        # there is a bug where on_episode_step gets called where it shouldn't
        env = base_env.get_unwrapped()[0]
        for agent, comm in env.communications.items():
            if max(comm) == 1:
                symbol = comm.index(1)
                self.comm_history[symbol] += 1

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs,):
        env = base_env.get_unwrapped()[0]
        for food, count in enumerate(env.num_exchanges):
            episode.custom_metrics[f"exchange_{food}"] = count
        for symbol, count in enumerate(self.comm_history):
            episode.custom_metrics[f"comm_{symbol}"] = count


if __name__ == "__main__":
    num_agents = 3
    env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 100}
    tenv = Trade(env_config)
    obs = tenv.reset()
    for i in range(100):
        actions = {}
        for agent in tenv.agents:
            tenv.render()
            print(f"Action for {agent}: ", end="")
            action = int(input())
            actions[agent] = action
        obss, rews, dones, infos = tenv.step(actions)
        if dones["__all__"]:
            print("game over")
            break
