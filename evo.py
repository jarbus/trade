import re
import numpy as np
import ray
from statistics import mean
from math import ceil
from dataclasses import dataclass
from itertools import product, islice, cycle
from typing import Dict, List, Tuple
from ray.tune.logger import pretty_print
from args import get_args
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.catalog import MODEL_DEFAULTS
from trade_v4 import Trade, TradeCallback, POLICY_MAPPING_FN
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes

import random
from DIRS import RESULTS_DIR

args = get_args()

pops = [[f"f{f}a{a}" for a in range(args.pop_size//args.food_types)] for f in range(args.food_types)]
env_config = {"window": (3, 3),
    "grid": (args.gx, args.gy),
    "food_types": 2,
    "latest_agent_ids": [(args.pop_size//args.food_types)-1 for _ in range(args.food_types)],
    "matchups": list(product(*pops)),
    "episode_length": args.episode_length,
    "move_coeff": args.move_coeff,
    "dist_coeff": args.dist_coeff,
    "ineq_coeff": args.ineq_coeff,
    "death_prob": args.death_prob,
    "twonn_coeff": args.twonn_coeff,
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
    "food_env_spawn": args.food_env_spawn,
    "day_night_cycle": args.day_night_cycle,
    "night_time_death_prob": args.night_time_death_prob,
    "day_steps": args.day_steps,
    "policy_mapping_fn": POLICY_MAPPING_FN,
    "vocab_size": 0}

test_env = Trade(env_config)
obs_space = test_env.observation_space
act_space = test_env.action_space

pol_config = {
    "model": {
        "conv_filters": [[128, [3, 3], 1], [128, [3, 3], 1], [128, [3, 3], 1]],
        "conv_activation": "relu",
        "post_fcnet_hiddens": [128, 128],
        "post_fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_cell_size": 512,
        "lstm_use_prev_action": False,
        "max_seq_len": 50,
    },
    "gamma": 0.99,
}
pol_spec = PolicySpec(None, obs_space, act_space, pol_config)


def policies_to_pops(policies:List[str]) -> List[List[str]]:
    pops = [[] for _ in range(args.food_types)]
    for pol in policies:
        try:
            f, a = re.match(r"f(\d+)a(\d+)", pol).groups()
            pops[int(f)].append(f"f{f}a{a}")
        except:
            continue
    for pop in pops:
        pop.sort()
    return pops


def all_vs_all(trainer, workerset):
    """This function took a while to make.
    - Don't try and configure individual envs on a
    worker, you can only sample one episode at a time.
    - Don't try and change a worker env after you've called
    it, changing the env setting is complicated."""
    pops = policies_to_pops(list(trainer.config["multiagent"]["policies"].keys()))
    matchups = list(product(*pops))
    workers = workerset.remote_workers()
    bin = ceil(len(matchups) / len(workers))
    worker_matchups = [matchups[m:m+bin] 
            for m in range(0, len(matchups), bin)]
    futures = []
    for i, w in enumerate(workers):
        print(f"Running {worker_matchups[i]} on worker_{i}...")
        w.foreach_env_with_context.remote(
                lambda env, ctx: env.set_matchups(worker_matchups[i]))
        for _ in worker_matchups[i]:
            futures.append(w.sample.remote())
    ray.get(futures)
    episodes, _ = collect_episodes( remote_workers=workerset.remote_workers(), timeout_seconds=99999)
    print(f"Collected {len(episodes)} episodes.")

    metrics = summarize_episodes(episodes)
    return metrics

policies = {pol: pol_spec for pop in pops for pol in pop}
policy_mapping_fn = POLICY_MAPPING_FN

env_name = "trade_v4"

batch_size = args.batch_size
config={
        # Environment specific
        "env": env_name,
        "env_config": env_config,
        "callbacks": TradeCallback,
        "recreate_failed_workers": True,
        "log_level": "ERROR",
        "framework": "torch",
        "horizon": args.episode_length * args.num_agents,
        "num_gpus": 0,
        "evaluation_config": {
            "env_config": {"agents": [] },
            "explore": False
        },
        "evaluation_duration": 20, # make this number of envs per worker
        "evaluation_num_workers" : 4,
        "num_workers": 1,
        "custom_eval_function": all_vs_all,

        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 20,
        "batch_mode": 'truncate_episodes',
        "lambda": 0.95,
        "gamma": .99,
        "model": pol_config["model"],
        "clip_param": 0.03,
        "entropy_coeff": 0.05,
        'vf_loss_coeff': 0.25,
        "num_sgd_iter": 5,
        "sgd_minibatch_size": batch_size,
        "train_batch_size": batch_size,
        'rollout_fragment_length': 50,
        'lr': 1e-04,
        # Method specific
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        }}


class ReusablePPOTrainer(ppo.PPOTrainer):
    def reset_config(self, new_config):
        self.setup(new_config)
        return True


pbt_interval = args.checkpoint_interval if args.pbt else 10_000_000_000_000


if __name__ == "__main__":

    if args.ip:
        ray.init(address=args.ip, _redis_password="longredispassword")


    register_env(env_name, lambda config: Trade(config))
    env = Trade(env_config)
    
    trainer = ReusablePPOTrainer(config=config, env=Trade)

    def add_pol(pop: int):
        config["env_config"]["latest_agent_ids"][pop] += 1
        pol_id = config["env_config"]["latest_agent_ids"][pop]
        pol_name = f"f{pop}a{pol_id}"

        print(f"Adding {pol_name}")
        config["multiagent"]["policies"][pol_name] = pol_spec
        trainer.add_policy(pol_name,
            PPOTorchPolicy,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(config["multiagent"]["policies"].keys()),
            config=config)
        return pol_name

    def rm_pol(pol_name: str):
        print(f"Removing {pol_name}")
        config["multiagent"]["policies"].pop(pol_name)
        trainer.remove_policy(pol_name,
                policy_mapping_fn=POLICY_MAPPING_FN,
                policies_to_train=list(config["multiagent"]["policies"].keys()))

    def mutate_weights(weights: dict):
        for d in weights.keys():
            weights[d] = weights[d] + np.random.normal(size=weights[d].shape)
        return weights

    def cp_and_mut(src_pol: str, dst_pol: str):
        print(f"Mutated copy: {src_pol}->{dst_pol}")
        trainer.set_weights({dst_pol: mutate_weights(trainer.get_weights([src_pol])[src_pol])})

    def sort_pops(rewards: Dict[str, int]) -> List[List[Tuple[float, str]]]:
        food_pols = [[] for _ in range(args.food_types)]
        for pol in rewards.keys():
            try:
                f, a = re.match(r"policy_f(\d+)a(\d+)_reward", pol).groups()
                food_pols[int(f)].append((mean(rewards[pol]), f"f{f}a{a}"))
            except:
                continue
        for f in range(len(food_pols)):
            food_pols[f].sort()
        return food_pols

    def selection(pops_rewards: List[List[Tuple[float, str]]]):
        new_pops_rewards = []
        for pop in pops_rewards:
            mid = len(pop)//2
            for pol in sorted(pop)[:mid]:
                rm_pol(pol[1])
            new_pops_rewards.append(pop[mid:])
        assert len(new_pops_rewards[-1]) == len(pops_rewards[-1])//2, "Selection did not halve the population."
        return new_pops_rewards

    def reproduction(pops_rewards: List[List[Tuple[float, str]]]):
        print("Beginning reproduction")
        new_pops = []
        for f, pop in enumerate(pops_rewards):
            pols = [pol[1] for pol in pop]
            new_pops.append(pols.copy())
            # Compute fitness-proportional selection probabilities
            rews = np.array([pol[0] for pol in pop])
            probs = rews - rews.min()
            if probs.sum() == 0:
                probs = np.array([1/len(probs) for _ in probs])
            else:
                probs = probs / probs.sum()
            # Create mutated copies of selected policies 
            # proportional to their fitness to double 
            # the population size
            for _ in range(len(pop)):
                src_pol = np.random.choice(pols, p=probs)
                new_pol = add_pol(f)
                cp_and_mut(src_pol, new_pol)
                new_pops[f].append(new_pol)

        return new_pops

    def evolve(trainer):
        evaluate_result = trainer.evaluate()
        eval_rewards = evaluate_result["evaluation"]["hist_stats"]
        sorted_pops = sort_pops(eval_rewards)
        selected_pops = selection(sorted_pops)
        new_pops = reproduction(selected_pops)
        matchups = list(product(*new_pops))
        config["env_config"]["matchups"] = matchups
        print(f"Setting new matchups: {matchups}")

        trainer.reset_config(config)

        for w in trainer.workers.remote_workers():
            w.foreach_env.remote(
                    lambda env: env.set_matchups(matchups))



    for i in range(20):
        print("Training")
        for j in range(10):
            trainer.train()
        # TODO:figure out why this is not returning num_env_episodes
        print("Evolve")
        evolve(trainer)
