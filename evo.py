import os
import sys
import re
import shutil
import numpy as np
import ray
import glob
import pickle
import copy
import pandas as pd
from datetime import datetime
from subprocess import Popen, run
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
#from ray.tune.logger import NoopLogger
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.catalog import MODEL_DEFAULTS
import models
from trade_v4 import Trade, TradeCallback, POLICY_MAPPING_FN
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.evaluation.metrics import collect_metrics

import random
from DIRS import RESULTS_DIR

class ReusablePPOTrainer(ppo.PPOTrainer):
    def reset_config(self, new_config):
        self.cleanup()
        self.setup(new_config)
        return True

def policies_to_pops(policies:List[str]) -> List[List[str]]:
    pops = [[] for _ in range(args.food_types)]
    for pol in policies:
        try:
            f, a = re.match(r"f(\d+)a(\d+)", pol).groups()
            pops[int(f)].append(f"f{f}a{a}")
        except Exception as e:
            continue
    for pop in pops:
        pop.sort()
    return pops


def get(_worker, _attr):
    return ray.get(_worker.foreach_env.remote( lambda env: getattr(env, _attr)))

def timestamp():
    return str(datetime.now().replace(microsecond=0))


def all_vs_all(trainer, workerset):
    """This function took a while to make.
    - Don't try and configure individual envs on a
    worker, you can only sample one episode at a time.
    - Don't try and change a worker env after you've called
    it, changing the env setting is complicated."""
    print(f"Running all_vs_all for timestamp {timestamp()}")
    pops = policies_to_pops(list(trainer.config["multiagent"]["policies"].keys()))
    matchups = [[a for pop in pops for a in pop]]
    workers = workerset.remote_workers()
    bin = ceil(len(matchups) / len(workers))
    worker_matchups = [matchups[m:m+bin] for m in range(0, len(matchups), bin)]
    futures = []
    agents_in_matchups = list(set([a for wm in worker_matchups for m in wm for a in m]))

    render_dir = os.path.join(EXP_DIR, f"outs-{timestamp()}")
    os.path.isdir(render_dir) or os.mkdir(render_dir)
    # Give each worker all matchups it needs to run
    for i, w in enumerate(workers):
        # print(f"Running {worker_matchups[i]} on worker_{i}...")
        if i >= len(worker_matchups):
            break
        ray.get(w.foreach_env.remote(
                lambda env: env.set_matchups(worker_matchups[i])))

        ray.get(w.foreach_env_with_context.remote(
                lambda env, ctx: env.set_render_path(render_dir+f"/{ctx.vector_index}-")))


        for _ in worker_matchups[i]:
            futures.append(w.sample.remote())


    batches = ray.get(futures)
    metrics = collect_metrics(workerset.local_worker(), workerset.remote_workers(), timeout_seconds=999999)
    os.listdir(render_dir) or os.rmdir(render_dir)
    #assert  == len(metrics["policy_reward_mean"])
    if len(metrics["policy_reward_mean"]) != len(agents_in_matchups):
        print(f"agents in matchup: {agents_in_matchups}")
        print(f"Returned: {metrics['policy_reward_mean']}")
    assert len(metrics["policy_reward_mean"]) == len(agents_in_matchups)
    assert len(metrics["policy_reward_mean"]) == len(trainer.config["multiagent"]["policies"])
    return metrics


def add_pol(pop: int):
    config["env_config"]["latest_agent_ids"][pop] += 1
    pol_id = config["env_config"]["latest_agent_ids"][pop]
    pol_name = f"f{pop}a{pol_id}"

    #print(f"Adding {pol_name}")
    config["multiagent"]["policies"][pol_name] = pol_spec
    trainer.add_policy(pol_name,
        PPOTorchPolicy,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=list(config["multiagent"]["policies"].keys()),
        config=config)
    return pol_name


def add_pol_by_name(pop: int, name: str):
    pop, a = re.match(r"f(\d+)a(\d+)", name).groups()
    config["env_config"]["latest_agent_ids"][int(pop)] = max(
        config["env_config"]["latest_agent_ids"][int(pop)], int(a))

    #print(f"Adding {pol_name}")
    config["multiagent"]["policies"][name] = pol_spec
    trainer.add_policy(name,
        PPOTorchPolicy,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=list(config["multiagent"]["policies"].keys()),
        config=config)
    return name

def rm_pol(pol_name: str):
    #print(f"Removing {pol_name}")
    config["multiagent"]["policies"].pop(pol_name)
    trainer.remove_policy(pol_name,
            policy_mapping_fn=POLICY_MAPPING_FN,
            policies_to_train=list(config["multiagent"]["policies"].keys()))

def add_row(df, result):
    new_row = pd.DataFrame(result["custom_metrics"], index=[1])
    for stat in ["min", "max", "mean"]:
        met = f"episode_reward_{stat}"
        new_row[met] = result[met]
    return pd.concat([df, new_row], ignore_index=True, sort=False)

def save(trainer, path):
    # make a list of all previous checkpoints
    prev_checks = []
    for f in glob.glob(f"{os.path.join(path, '*')}"):
        if re.match(".*checkpoint-(\d+)", f):
            prev_checks.append(f)
    
    trainer.save_checkpoint(path)
    with open(os.path.join(path, "policies.p"), "wb") as f:
        pickle.dump(list(trainer.config["multiagent"]["policies"].keys()), f)
    # delete previous checkpoints
    for f in prev_checks:
        os.remove(f)

def load(trainer, path):
    print("Loading checkpoint from", path)
    global policies, results_df, RESULT_FILE
    results_df = pd.DataFrame()
    if os.path.exists(RESULT_FILE):
        results_df = pd.read_csv(RESULT_FILE)

    check = -1
    newest_checkpoint = ""
    for f in glob.glob(f"{os.path.join(path, '*')}"):
        m = re.match(".*checkpoint-(\d+)", f)
        if m:
            new_check = int(m.groups()[0])
            if new_check > check:
                check = new_check
                newest_checkpoint = f
    check_path = os.path.join(path, newest_checkpoint)

    if newest_checkpoint:

        assert new_check > 0
        print(f"Restoring from {check_path}")
        for pol in policies.copy().keys():
            print(f"Removing {pol}")
            rm_pol(pol)
        with open(os.path.join(path,"policies.p"), "rb") as f:
            saved_pols = pickle.load(f)
            assert len(saved_pols) > 1
            for pol in saved_pols:
                print(f"Adding {pol}")
                add_pol_by_name(trainer, pol)

        trainer.reset_config(config)
        trainer.load_checkpoint(check_path)
        trainer._iteration = new_check
        loaded_pops = policies_to_pops(saved_pols)
        assert all(len(pop) > 0 for pop in loaded_pops)
        matchups = [[a for pop in pops for a in pop]]
        assert len(matchups) > 0
        print("Setting matchups", matchups)

        for w in trainer.workers.remote_workers():
            w.foreach_env.remote(
                lambda env: env.set_matchups(matchups))


if __name__ == "__main__":

    args = get_args()
    CLASS_DIR = os.path.join(RESULTS_DIR, f"{args.class_name}")
    os.path.exists(CLASS_DIR) or os.mkdir(CLASS_DIR)

    EXP_DIR = os.path.join(CLASS_DIR, f"{args.name}")
    os.path.exists(EXP_DIR) or os.mkdir(EXP_DIR)

    RESULT_FILE=os.path.join(EXP_DIR,"results.txt")
    EVAL_FILE = os.path.join(EXP_DIR, "eval.csv")



    pops = [[f"f{f}a{a}" for a in range(args.pop_size//args.food_types)] for f in range(args.food_types)]
    matchups = [[a for pop in pops for a in pop]]
    env_config = {"window": (args.window, args.window),
        "grid": (args.gx, args.gy),
        "food_types": 2,
        "latest_agent_ids": [(args.pop_size//args.food_types)-1 for _ in range(args.food_types)],
        "matchups": matchups,
        "episode_length": args.episode_length,
        "fire_radius": args.fire_radius,
        "move_coeff": args.move_coeff,
        "no_multiplier": args.no_multiplier,
        "num_piles": args.num_piles,
        "dist_coeff": args.dist_coeff,
        "caps": args.caps,
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
        "model": getattr(models, args.model),
        "gamma": 0.99,
    }
    pol_spec = PolicySpec(None, obs_space, act_space, pol_config)

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
            "num_gpus": 1,
            "evaluation_config": {
                "env_config": {"agents": matchups[0]},
                "explore": False
            },
            "evaluation_duration": 10, # make this number of envs per worker
            "evaluation_num_workers" : 4,
            "num_workers": 4,
            "custom_eval_function": all_vs_all,

            "num_cpus_per_worker": 2,
            "num_envs_per_worker": 10,
            "batch_mode": 'truncate_episodes',
            "lambda": 0.95,
            "gamma": .99,
            "model": pol_config["model"],
            "clip_param": 0.03,
            "entropy_coeff": args.entropy_coeff,
            'vf_loss_coeff': 0.25,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": batch_size,
            "train_batch_size": batch_size,
            'rollout_fragment_length': 50,
            'lr': args.learning_rate,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "logger_config": {
                "type": "ray.tune.logger.NoopLogger",

            }}

    pbt_interval = args.checkpoint_interval if args.pbt else 10_000_000_000_000
    print(config)
 


    print("SPAWNING RAY")
    if args.ip:
        ray.init(address=args.ip, _redis_password="longredispassword")

    print("REGISTER_ENV")
    register_env(env_name, lambda config: Trade(config))
    env = Trade(env_config)
    
    print("MAKE TRAINER")
    trainer = ReusablePPOTrainer(config=config, env=Trade)

    print("BEGINNING LOOP")
    load(trainer, EXP_DIR)

    def run_training():
        global results_df
        prev_result = {'custom_metrics': {}}
        i = 0
        while True:
            i += 1
            for j in range(40*(args.pop_size)):
                print("Training")
                result = trainer.train()
                print("Trained")
                # trainer.train() returns a result with the same custom_metric dict
                # if no new episodes were completed. in this case, skip logging.
                if not result["custom_metrics"] or  result["custom_metrics"] == prev_result['custom_metrics']:
                    print("Skipping no new results")
                    continue

                print("Writing results")
                result["custom_metrics"]["step"] = trainer._iteration
                results_df = add_row(results_df, result)
                # we remove the step from the custom_metrics dict so that
                # we can skip equal results
                del result["custom_metrics"]["step"]
                prev_result = result

            tmp_result_file = RESULT_FILE+"-tmp"
            results_df.round(2).to_csv(tmp_result_file, index=False)
            run(f"mv -f {tmp_result_file} {RESULT_FILE}".split())

            eval_mets = trainer.evaluate()
            eval_mets["evaluation"]["custom_metrics"]["step"] = trainer._iteration
            eval_df = pd.DataFrame.from_dict(eval_mets['evaluation']["custom_metrics"], orient="index").T
            eval_df.round(2).to_csv(EVAL_FILE, mode='a', header=(not os.path.exists(EVAL_FILE)), index=False)
            print(eval_mets)

            if i % 20 == 0:
                print(f"Saving trainer for timestamp {timestamp()}")
                save(trainer, EXP_DIR)


    def run_interactive():
        obss = test_env.reset()
        test_env.render_interactive()
        states = {}
        for agent in test_env.agents:
            policy = trainer.get_policy(agent)
            states[agent] = policy.get_initial_state()

        for i in range(400):
            actions = {}
            for agent in obss.keys():
                policy = trainer.get_policy(agent)
                actions[agent], states[agent], logits = policy.compute_single_action(obs=np.array(obss[agent]), state=states[agent], policy_id=agent)
                # override agent action with custom input if specified
                act = input(f'act for {test_env.agents.index(agent)}:')
                # only process first char
                if len(act) > 1:
                    act = act[0]
                # map udlr to vim keys
                vim="k j h l".split()
                if act in vim:
                    act = vim.index(act)
                if act != "":
                    actions[agent] = int(act)

            obss, rews, dones, infos = test_env.step({agent: action for agent, action in actions.items() if not test_env.compute_done(agent)})
            test_env.render_interactive()
            if dones["__all__"]:
                print("--------FINAL-STEP--------")
                #test_env.render()
                print("game over")
                break
    if args.interactive:
        run_interactive()
    else:
        run_training()
