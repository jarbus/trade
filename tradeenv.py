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
        food_types = env_config.get("food_types", 2)
        num_agents = env_config.get("num_agents", 2)
        episode_length = env_config.get("episode_length", 100)
        super().__init__()
        self.food_types = food_types
        self.num_inputs = food_types * (num_agents+1)
        self.agent_food_counts = dict()
        self.max_steps = episode_length
        self.MOVES = []
        for f in range(food_types):
            self.MOVES.extend([f"PICK_{f}", f"PLACE_{f}"])
        self.MOVES.append("NONE")
        self.num_actions = len(self.MOVES)
        self.num_exchanges = []
        self.exchange_table = []

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

    def compute_observation(self, agent=None):
        # stores action of current agent
        curr_state = self.table.copy()
        if agent:
            curr_state.extend([*self.agent_food_counts[agent]])
        for other_agent in self.agents:
            if agent != other_agent:
                curr_state.extend(self.agent_food_counts[other_agent])
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
        for (agent, action) in actions.items():
            if action != self.num_actions - 1:
                pick = (action % 2 == 0)
                food = floor(action / 2)
                if pick:
                    self.agent_food_counts[agent][food] += self.table[food]
                    self.table[food] = 0
                    self.num_exchanges[food] += self.compute_exchange_amount(food, agent)
                    self.exchange_table[food] = defaultdict(int)
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    self.agent_food_counts[agent][food] -= PLACE_AMOUNT
                    if self.agent_food_counts[agent][food] < 0:
                        print(self.agent_food_counts)
                    self.table[food] += PLACE_AMOUNT
                    self.exchange_table[food][agent] += PLACE_AMOUNT

        for agent in self.agents:
            self.agent_food_counts[agent] = [x - 0.1 for x in self.agent_food_counts[agent]]

        self.steps += 1

        obs = {agent: self.compute_observation(agent) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        rewards = {agent: self.compute_reward(agent) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        infos = {}
        return obs, rewards, dones, infos


if __name__ == "__main__":
    tenv = Trade({})
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

class TradeCallback(DefaultCallbacks):

    def on_episode_end(
        self,
        worker,  #: RolloutWorker,
        base_env,  #: BaseEnv,
        policies,  #: Dict[str, Policy],
        episode,  #: MultiAgentEpisode,
        **kwargs,
    ):
        num_exchanges = base_env.get_unwrapped()[0].num_exchanges
        for food, count in enumerate(num_exchanges):
            episode.custom_metrics[f"exchange_{food}"] = count
