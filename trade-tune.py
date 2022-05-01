import torch
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from tradeenv import Trade, TradeCallback
from ray.tune.schedulers import PopulationBasedTraining
import random

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 1000),
        "train_batch_size": lambda: random.randint(2000, 160000),
    },
    custom_explore_fn=explore,
)


if __name__ == "__main__":

    env_name = "trade_v1"

    register_env(env_name, lambda config: Trade(config))

    num_agents = 3
    env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 100, "vocab_size": 0}

    test_env = Trade(env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            # "agent_id": i,
            "model": {
                # Change individual keys in that dict by overriding them, e.g.
                "fcnet_hiddens": [64, 64, 64],
                "fcnet_activation": "relu",
                "use_lstm": True,
            },
            "gamma": 0.99,
        }
        return PolicySpec(None, obs_space, act_space, config)

    # policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
    policy = gen_policy(0)
    policies = {f"player_{a}": policy for a in range(num_agents)}
    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO",
        scheduler=pbt,
        metric="episode_reward_mean",
        mode="max",
        num_samples=10,
        stop={"timesteps_total": 100_000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            "env_config": env_config,
            "callbacks": TradeCallback,
            # General
            "log_level": "ERROR",
            "framework": "torch",
            "num_gpus": 0.75,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',
            # 'use_critic': True,
            'use_gae': True,
            "lambda": 0.9,
            "gamma": .99,
            # "kl_coeff": 0.001,
            # "kl_target": 1000.,
            "clip_param": tune.choice([0.1, 0.2, 0.3, 0.4]),
            'grad_clip': None,
            "entropy_coeff": 0.2,
            'vf_loss_coeff': 0.25,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([100, 500, 1000]),
            "train_batch_size": tune.choice([1000, 2000, 4000]),
            'rollout_fragment_length': 128,
            'lr': 2e-05,
            "clip_actions": True,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda aid, **kwargs: aid),
            }})
