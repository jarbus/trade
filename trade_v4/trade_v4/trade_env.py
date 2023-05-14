import os
import sys
import numpy as np
from math import floor
from itertools import cycle
from collections import defaultdict
from typing import List, Tuple, Dict
from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .light import Light
from .utils import add_tup, directions, valid_pos, inv_dist, punish_region, matchup_shuffler
from .spawners import FireCornerSpawner, FoodSpawner, DiscreteFoodSpawner, AgentSpawner

class TradeMetricCollector():
    def __init__(self, env):
        self.rew_base_health          = 0
        self.rew_shared_health        = 0
        self.rew_nn                   = 0
        self.rew_twonn                = 0
        self.rew_other_survival_bonus = 0
        self.rew_pun                  = 0
        self.rew_mov                  = 0
        self.rew_light                = 0
        self.rew_acts                 = 0
        self.rew_ineq                 = 0

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
        self.picked_counts[agent][food] += env.compute_pick_amount(x, y, food, agent)
        pass

    def collect_rew(self, env, base_health, shared_health, nn, twonn, other_survival_bonus, pun_rew, mov_rew, light, action_rewards, ineq):
        self.rew_base_health          += base_health
        self.rew_shared_health        += shared_health
        self.rew_nn                   += nn
        self.rew_twonn                += twonn
        self.rew_other_survival_bonus += other_survival_bonus
        self.rew_pun                  += pun_rew
        self.rew_mov                  += mov_rew
        self.rew_light                += light
        self.rew_acts                 += action_rewards
        self.rew_ineq                 += ineq

METABOLISM=0.1
PLACE_AMOUNT = 0.5
SCALE_DOWN = 30

NUM_ITERS = 100
ndir = len(directions)

class Trade(MultiAgentEnv):

    def __init__(self, env_config):
        #print(f"Creating Trade environment {env_config}")
        gx, gy = self.grid_size    = env_config.get("grid", (7, 7))
        self.food_types            = env_config.get("food_types", 2)
        #num_agents                 = env_config.get("num_agents", 2)
        self.matchups           = env_config.get("matchups")
        self.max_steps             = env_config.get("episode_length", 100)
        self.vocab_size            = env_config.get("vocab_size", 0)
        self.window_size           = env_config.get("window", (3, 3))
        self.dist_coeff            = env_config.get("dist_coeff", 0.0)
        self.move_coeff            = env_config.get("move_coeff", 0.0)
        self.death_prob            = env_config.get("death_prob", 0.1)
        self.fire_radius           = env_config.get("fire_radius")
        self.caps                  = env_config.get("caps")
        self.num_piles             = env_config.get("num_piles", 5)
        self.day_night_cycle       = env_config.get("day_night_cycle", False)
        self.day_steps             = env_config.get("day_steps", 20)
        self.fires                 = env_config.get("fires")
        self.foods                 = env_config.get("foods")
        self.night_time_death_prob = env_config.get("night_time_death_prob", 0.1)
        self.punish                = env_config.get("punish", False)
        self.render_ep             = env_config.get("render", False)
        self.punish_coeff          = env_config.get("punish_coeff", 3)
        self.survival_bonus        = env_config.get("survival_bonus", 0.0)
        self.respawn               = env_config.get("respawn", False)
        self.twonn_coeff           = env_config.get("twonn_coeff", 0.0)
        self.ineq_coeff            = env_config.get("ineq_coeff", 0.0)
        self.light_coeff           = env_config.get("light_coeff", 1.0)
        self.pickup_coeff          = env_config.get("pickup_coeff", 1.0)
        self.health_baseline       = env_config.get("health_baseline", False)
        self.policy_mapping_fn     = env_config.get("policy_mapping_fn")
        self.food_env_spawn        = env_config.get("food_env_spawn")
        self.no_multiplier         = env_config.get("no_multiplier")
        self.food_agent_start      = env_config.get("food_agent_start", 0)
        self.share_health          = env_config.get("share_health")
        self.padded_grid_size      = add_tup(self.grid_size, add_tup(self.window_size, self.window_size))
        self.light                 = Light(self.grid_size, self.fires, 2/self.day_steps, self.fire_radius)

        super().__init__()

        self.render_path = None


        # (self + policies) * (food frames and pos frame)
        food_frame_and_agent_channels = (2) * (self.food_types+1)
        # agents_and_foods + food frames + comms
        self.channels = food_frame_and_agent_channels + (self.food_types) + (self.vocab_size) + int(self.punish) + int(self.day_night_cycle)
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

        # Get rid of food type indicator for the spawner
        food_centers = [(fc[1], fc[2]) for fc in self.foods]
        self.food_spawner = FoodSpawner(self.grid_size, food_centers) 
        self.food_spawner = DiscreteFoodSpawner(self.grid_size, food_centers) 
        self.agent_spawner = FireCornerSpawner(self.grid_size, self.fires)

        self.action_space = Discrete(self.num_actions)
        self.obs_size = (*add_tup(add_tup(self.window_size, self.window_size), (1, 1)), self.channels)
        self.observation_space = Box(low=np.full(self.obs_size, -1, dtype=np.float32), high=np.full(self.obs_size, 10))
        self._skip_env_checking = True
        self.matchup_iterator = matchup_shuffler(self.matchups)

    def set_matchups(self, matchups: List[Tuple[str, str]]):
        self.matchups = matchups
        self.matchup_iterator = matchup_shuffler(self.matchups)
        #self.reset()

    def set_render_path(self, path:str):
        self.render_path = path
        # deterministic order for rendering matchups
        # ray needs to get last obs, which means reset will be called
        # on the last step. Therefore, next() needs to return something
        # at the end, so we make matchups cycle.
        self.matchup_iterator = cycle(self.matchups)
        #print(f"Render path set to {self.render_path}")
        #print(f"Matchups set to {self.matchups} in that order")
        #self.reset()

    def food_multiplier(self, agent: str, food: int):
        if self.no_multiplier:
            return 1
        return 1 if int(agent[1]) == food else 0.5

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def generate_food(self):
        fc = self.food_env_spawn if self.respawn else 10
        #food_counts = [(0, fc), (0, fc), (1, fc), (1, fc)]
        food_counts = [(f[0], fc) for f in self.foods]
        self.table[:,:,:,:] = 0

        for i in range(self.num_piles):
            spawn_spots = self.food_spawner.gen_poses()
            for spawn_spot, (ft, fc) in zip(spawn_spots, food_counts):
                fx, fy = spawn_spot
                self.table[fx, fy, ft, len(self.agents)] += fc / self.num_piles

    def reset(self):

        if self.matchups == [] or self.matchups == [()]:
            return {}


        self.light.reset()
        self.agents = list(next(self.matchup_iterator)) # self.possible_agents[:]


        if not self.agents:
            raise ValueError(f"Agents not set, matchup iterator returned False, matchups={self.matchups}")
        #print(f"Running env with agents {self.agents}")
        # print(f"resetting env with {self.agents}")
        self.action_rewards = {a: 0 for a in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.moved_last_turn = {agent: False for agent in self.agents}
        gx, gy = self.grid_size
        # use the last slot in the agents dimension to specify naturally spawning food
        self.table = np.zeros((*self.grid_size, self.food_types, len(self.agents)+1), dtype=np.float32)

        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        self.generate_food()
        #self.agent_spawner.reset()
        spawn_spots = self.agent_spawner.gen_poses(n=len(self.agents))
        self.agent_positions = {agent: spawn_spot for agent, spawn_spot in zip(self.agents, spawn_spots)}
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        # assumption: agent names follow a pattern fxay where x is a food type and y is an agent number
        # when agents start with a resource, we use x to determine what resource they start with
        # this assumption reduces lots of complexity elsewhere
        self.agent_food_counts = {agent: [self.food_agent_start * float(agent[1]==str(f)) for f in range(self.food_types) ] for agent in self.agents}
        self.mc = TradeMetricCollector(self)
        # keep old copy of player exchanges
        self.player_exchanges = self.mc.player_exchanges.copy()  
        return {self.agents[0]: self.compute_observation(self.agents[0])}

    def render(self, mode="human", out=sys.stdout):

        out.write(f"--------STEP-{self.steps}--------\n")
        for agent in self.agents:
            out.write(f"{agent}: {self.agent_positions[agent]} {[round(fc, 2) for fc in self.agent_food_counts[agent]]} {self.compute_done(agent)}\n")
        for food in range(self.food_types):
            out.write(f"food{food}:\n")
            for row in self.table[:,:,food].sum(axis=2).round(2):
                out.write(str(list(row)).replace(",","")+"\n")
        if self.day_night_cycle:
            out.write(f"Light:\n")
            for row in self.light.frame.round(2):
                out.write(str(list(row)).replace(",","")+"\n")
        for (agent1, agent2, food_type), count in self.player_exchanges.items():
            new_count = self.mc.player_exchanges[(agent1, agent2, food_type)]
            if new_count != count:
                assert new_count > count
                out.write(f"Exchange: {agent1} gave {new_count-count} of food {food_type} to {agent2}\n")
        out.write(f"Total exchanged so far: {self.mc.num_exchanges}\n")
        for agent, comm in self.communications.items():
            if comm and max(comm) >= 1:
                out.write(f"{agent} said {comm.index(1)}\n")
        self.player_exchanges = self.mc.player_exchanges.copy()

    def render_interactive(self, out=sys.stdout):
        """displays lossy text view that's easier to interact with"""
        grid = np.full((self.grid_size[0], self.grid_size[1]), ".", dtype='<U7')
        for i, agent in enumerate(self.agents):
            ax, ay = self.agent_positions[agent]
            if grid[ax, ay] == ".":
                grid[ax, ay] = f"{i}"
            else:
                grid[ax, ay] = grid[ax, ay] + f"{i}"

        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                for f in range(self.food_types):
                    if np.sum(self.table[r, c, f, :]) <= 0:
                        continue
                    foodtype = ["f", "g"][f]
                    if grid[r, c] == ".":
                        grid[r, c] = foodtype
                    else:
                        grid[r, c] = grid[r, c] + foodtype
        out.write(f"--------STEP-{self.steps}--------\n")
        vim="k j h l".split()
        for i, move in enumerate(self.MOVES):
            if i <= 3:
                chr = vim[i]
            else:
                chr = i

            out.write(f"{chr}: {move} ")
        out.write(f"\n")
        for agent in self.agents:
            out.write(f"{agent}: {self.agent_positions[agent]} {[round(fc, 2) for fc in self.agent_food_counts[agent]]} {self.compute_done(agent)}\n")
        for grid_row in grid:
            spots = [(3-len(c))*" "+c if len(c) < 3 else c for c in grid_row]
            out.write("".join(spots)+"\n")

    def compute_observation(self, agent):
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
        pol_pos_frames   = np.zeros(self.grid_size, dtype=np.float32)
        pol_food_frames  = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        for i, a in enumerate(self.agents):
            if self.compute_done(a):
                continue
            oax, oay = self.agent_positions[a]
            comm_frames[:, oax, oay] = self.communications[a]

            if a != agent:
                pol_pos_frames[oax, oay] += 1
                pol_food_frames[:, oax, oay] += self.agent_food_counts[a]
            else:
                self_food_frames[:,  oax, oay] += self.agent_food_counts[a]

        pol_frames = np.stack([*pol_food_frames, pol_pos_frames])
        agent_and_food_frames = np.stack([*pol_frames, self_pos_frame, *self_food_frames])

        if self.punish:
            pun_frames = np.sum(self.punish_frames, axis=0)[None, :, :]
        else:
            pun_frames = np.zeros((0, *self.grid_size), dtype=np.float32)

        if self.day_night_cycle:
            light_frames = self.light.frame[None, :, :] * SCALE_DOWN
        else:
            light_frames = np.zeros((0, *self.grid_size), dtype=np.float32)


        xpos_frame = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gx
        ypos_frame = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gy

        frames = np.stack([*food_frames, *agent_and_food_frames, *light_frames], axis=2)
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

        base_health = 0
        shared_health = 0
        ineq = 0
        if self.health_baseline:
            num_of_food_types = sum(1 for f in self.agent_food_counts[agent] if f >= 0.1)
            base_health = [0, 0.1, 1][num_of_food_types]
            for a in self.agents:
                if a != agent:
                    num_of_food_types = sum(1 for f in self.agent_food_counts[a] if f >= 0.1)
                    other_health = [0, 0.1, 1][num_of_food_types]
                    shared_health += other_health * self.share_health
                    ineq += max(other_health - base_health, 0)

        else:
            base_health = 1
        assert ineq >= 0

        light_rew = 0 if self.light.contains(self.agent_positions[agent]) else self.light_coeff * self.light.frame[self.agent_positions[agent]]

        nn_rew    =  (self.dist_coeff * dists[-1])
        twonn_rew = -(self.twonn_coeff * dists[-2])
        pun_rew   = -self.punish_coeff * punishment
        mov_rew   = -self.move_coeff * int(self.moved_last_turn[agent])
        act_rew   = self.pickup_coeff * self.action_rewards[agent]
        ineq_rew  = -self.ineq_coeff * ineq

        # Remember to update this function whenever you add a new reward
        self.mc.collect_rew(self, base_health, shared_health, nn_rew, twonn_rew, other_survival_bonus, pun_rew, mov_rew, light_rew, act_rew, ineq_rew)
        assert light_rew <= 0
        assert base_health >= 0
        assert act_rew >= 0

        rew  = base_health + light_rew + act_rew
        return rew

    def compute_exchange_amount(self, x: int, y: int, food: int, picker: int):
        return sum(count for a, count in enumerate(self.table[x][y][food]) if a != picker and a != len(self.agents))

    def compute_pick_amount(self, x: int, y: int, food: int, agent: str):
        # compute spawned food to be collected
        env_food = self.table[x][y][food][len(self.agents)] 
        assert env_food >= 0
        # multiply by agent's food multiplier
        max_pick_food = env_food * self.food_multiplier(agent, food)
        assert max_pick_food >= 0
        # compute capped food
        agent_food = self.agent_food_counts[agent][food]
        assert agent_food >= 0
        assert agent_food <= self.caps[food]
        capped_food = min(agent_food + max_pick_food, self.caps[food])
        assert capped_food >= 0
        # compute amount to be picked
        picked_food = capped_food - agent_food
        assert picked_food >= 0
        return picked_food

    def compute_collect_amount(self, x: int, y: int, food: int, agent: str):
        # compute dropped food to be collected
        dropped_food = self.table[x][y][food][:-1].sum()
        # compute capped food
        agent_food = self.agent_food_counts[agent][food]
        capped_food = min(agent_food + dropped_food, self.caps[food])
        # compute amount to be collected
        collected_food = capped_food - agent_food
        return collected_food



    def update_dones(self):
        for agent in self.agents:
            if self.compute_done(agent):
                continue
            self.dones[agent] = self.steps >= self.max_steps
    def next_agent(self, agent):
        return self.agents[(self.agents.index(agent)+1) % len(self.agents)]

    def step(self, actions):

        # placed goods will not be available until next turn
        for agent in actions.keys():
            if agent not in self.agents:
                raise ValueError(f"Received action for {agent} which is not in {self.agents}")
        gx, gy = self.grid_size
        for agent, action in actions.items():
            self.action_rewards[agent] = 0
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
                    ###########
                    # Forage from env
                    ###########
                    # Compute pick metrics before performing pick
                    self.mc.collect_pick(self, agent, x, y, food, aid)
                    # Compute amount foraged from env, includes multiplier
                    env_food_picked = self.compute_pick_amount(x, y, food, agent)
                    # Remove from env, ignore multiplier
                    env_food_removed = env_food_picked / self.food_multiplier(agent, food)
                    self.table[x, y, food, len(self.agents)] -= env_food_removed
                    # handle minor floating point errors
                    if -0.01 < self.table[x, y, food, len(self.agents)] < 0:
                        self.table[x, y, food, len(self.agents)] = 0
                    # throw exception if negative table value not a result of minor error
                    elif self.table[x, y, food, len(self.agents)] <= -0.01:
                        raise ValueError(f"Table entry is negative: {self.table[x, y, food, len(self.agents)]}")
                    # Update food counts
                    self.agent_food_counts[agent][food] += env_food_picked
                    assert self.agent_food_counts[agent][food] <= self.caps[food]
                    # Update pickup reward
                    self.action_rewards[agent] += env_food_picked

                    ######################################
                    # Collect from agents
                    ######################################
                    dropped_food = self.compute_collect_amount(x, y, food, agent)
                    # Update agent food counts
                    self.agent_food_counts[agent][food] += dropped_food
                    # Clear food from table
                    for i in range(len(self.agents)):
                        amt_to_rm = min(self.table[x, y, food, i], dropped_food)
                        dropped_food -= amt_to_rm
                        self.table[x, y, food, i] -= amt_to_rm
                        # all food has been cleared
                        if dropped_food <= 0:
                            break

                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    actual_place_amount = PLACE_AMOUNT
                    self.agent_food_counts[agent][food] -= actual_place_amount
                    self.table[x, y, food, aid] += actual_place_amount
                    self.mc.collect_place(self, agent, food, actual_place_amount)
            # last action is noop
            elif action in range(4 + self.food_types * 2 + int(self.punish), self.num_actions-1):
                symbol = action - (self.food_types * 2) - 4
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1
            self.agent_food_counts[agent] = [np.clip(x - METABOLISM, 0, c) for (c,x) in zip(self.caps, self.agent_food_counts[agent])]


            if agent == self.agents[-1]:
                self.steps += 1
                if self.render_path:
                    path = self.render_path + "-".join(self.agents)+".out"
                    with open(path, "a") as f:
                        self.render(out=f)
                self.light.step_light()
                # Once agents complete all actions, add placed food to table
                if self.respawn and self.light.dawn():
                    self.generate_food()

                    self.update_dones()

        obs = {self.next_agent(agent): self.compute_observation(self.next_agent(agent)) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        # self.mc.collect_lifetimes(dones)
        rewards = {self.next_agent(agent): self.compute_reward(self.next_agent(agent)) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        infos = {}
        return obs, rewards, dones, infos
