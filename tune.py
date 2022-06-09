import torch
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v3 import Trade, TradeCallback
from ray.tune.schedulers import PopulationBasedTraining
import random
import argparse

parser = argparse.ArgumentParser(description='Execute Trading Environment.')
parser.add_argument('--ip', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--random-start', action="store_true")
parser.add_argument('--respawn', action="store_true")
args = parser.parse_args()



def generate_configs():
    num_agents = 4
# env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 20, "vocab_size": 0}

    env_config = {"window": (3, 3),
                  "grid": (5, 5),
                  "food_types": 2,
                  "num_agents": num_agents,
                  "episode_length": 200,
                  "move_coeff": 0.0,
                  "dist_coeff": 1.0,
                  "death_prob": 0.1,
                  "random_start": args.random_start,
                  "respawn": args.respawn,
                  "vocab_size": 0}

    test_env = Trade(env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "model": {
                # Change individual keys in that dict by overriding them, e.g.
                "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1]],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [64, 64],
                "post_fcnet_activation": "relu",
                "use_lstm": True,
                "max_seq_len": 200,
            },
            "gamma": 0.99,
        }
        return PolicySpec(None, obs_space, act_space, config)

    # policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
    # policy = gen_policy(0)
    policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
    return env_config, policies 




# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    #if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
    config["sgd_minibatch_size"] = config["train_batch_size"]
    # ensure we run at least one sgd iter
    #if config["num_sgd_iter"] < 1:
    #    config["num_sgd_iter"] = 1
    return config

class ReusablePPOTrainer(ppo.PPOTrainer):
    def reset_config(self, new_config):
        self.setup(new_config)
        return True


pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=500,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.2),
        "entropy_coeff": lambda: random.uniform(0.01, 0.2),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 10),
        "train_batch_size": [1000, 2000, 4000],
    },
    custom_explore_fn=explore,
)


if __name__ == "__main__":

    if args.ip:
        ray.init(address=args.ip, _redis_password="longredispassword")

    env_name = "trade_v3"

    register_env(env_name, lambda config: Trade(config))

    env_config, policies = generate_configs()
    batch_size = 1000

    tune.run(
        ReusablePPOTrainer,
        name=f"random-starts",
        scheduler=pbt,
        metric="episode_reward_mean",
        mode="max",
        resume=False,
        num_samples=16,
        stop={"timesteps_total": 10_000_000},
        checkpoint_freq=500,
        reuse_actors=True,
        #local_dir="~/ray_results/"+env_name,
        local_dir="/work/garbus/ray_results/2-only",
        config={
            # Environment specific
            "env": env_name,
            "env_config": env_config,
            "callbacks": TradeCallback,
            # General
            "log_level": "ERROR",
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            "num_cpus_for_driver": 1,
            "num_envs_per_worker": 20,
            "batch_mode": 'truncate_episodes',
            "lambda": 0.95,
            "gamma": .99,
            "clip_param": 0.1,
            "entropy_coeff": 0.05,
            'vf_loss_coeff': 0.25,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": 5,
            "sgd_minibatch_size": batch_size,
            "train_batch_size": batch_size,
            'rollout_fragment_length': 100,
            'lr': tune.choice([1e-03]),
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda aid, **kwargs: aid),
            }})
