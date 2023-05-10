import numpy as np
import sys
from args import get_args
from trade_v4 import Trade, METABOLISM, PLACE_AMOUNT, POLICY_MAPPING_FN
import random
def act(env, acts):
    return {agent: env.MOVES.index(act) for agent, act in acts.items()}

def random_actions(env):
    return {agent: random.randint(0, len(env.MOVES)-1) for agent in env.agents if not env.compute_done(agent)}
args = get_args()

pops = [[f"f{f}a{a}" for a in range(args.pop_size//args.food_types)] for f in range(args.food_types)]
matchups = [[a for pop in pops for a in pop]]
env_config = {"window": (args.window, args.window),
    "grid": (args.gx, args.gy),
    "food_types": 2,
    "latest_agent_ids": [(args.pop_size//args.food_types)-1 for _ in range(args.food_types)],
    "matchups": matchups,
    "episode_length": args.episode_length,
    "move_coeff": args.move_coeff,
    "dist_coeff": args.dist_coeff,
    "fire_radius": args.fire_radius,
    "num_piles": args.num_piles,
    "no_multiplier": args.no_multiplier,
    "pickup_coeff": args.pickup_coeff,
    "share_health": args.share_health,
    "respawn": args.respawn,
    "fires": [(args.fires[i], args.fires[i+1]) for i in range(0, len(args.fires), 2)],
    "foods": [(*args.foods[i:i+3],) for i in range(0, len(args.foods), 3)],
    "survival_bonus": args.survival_bonus,
    "health_baseline": args.health_baseline,
    "punish": args.punish,
    "spawn_agents": args.spawn_agents,
    "spawn_food": args.spawn_food,
    "light_coeff": args.light_coeff,
    "punish_coeff": args.punish_coeff,
    "food_agent_start": args.food_agent_start,
    "caps": args.caps,
    "food_env_spawn": args.food_env_spawn,
    "day_night_cycle": args.day_night_cycle,
    "night_time_death_prob": args.night_time_death_prob,
    "day_steps": args.day_steps,
    "policy_mapping_fn": POLICY_MAPPING_FN,
    "vocab_size": 0}

test_env = Trade(env_config)
test_env.reset()
test_env.render()
test_env.reset()
# print(test_env.agent_positions)
# print(test_env.agent_positions)
