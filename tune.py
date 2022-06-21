import ray
from args import get_args
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v4 import Trade, TradeCallback
from ray.tune.schedulers import PopulationBasedTraining
import random
from DIRS import RESULTS_DIR



args = get_args()

def generate_configs():
    num_agents = 4
# env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 20, "vocab_size": 0}

    env_config = {"window": (3, 3),
                  "grid": (args.gx, args.gy),
                  "food_types": 2,
                  "num_agents": num_agents,
                  "episode_length": 200,
                  "move_coeff": args.move_coeff,
                  "dist_coeff": args.dist_coeff,
                  "death_prob": args.death_prob,
                  "twonn_coeff": args.twonn_coeff,
                  "respawn": args.respawn,
                  "survival_bonus": args.survival_bonus,
                  "health_baseline": args.health_baseline,
                  "punish": args.punish,
                  "spawn_agents": args.spawn_agents,
                  "punish_coeff": args.punish_coeff,
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
                "lstm_use_prev_action": False,
                "max_seq_len": 50,
            },
            "gamma": 0.99,
        }
        return PolicySpec(None, obs_space, act_space, config)

    # policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
    if args.num_policies == 1:
        policy = gen_policy(0)
        policies = {"pol1": policy}
        def policy_mapping_fn(aid, **kwargs):
            return "pol1"
    elif args.num_policies == 2:
        pol1 = gen_policy(0)
        pol2 = gen_policy(1)
        policies = {"pol1": pol1, "pol2": pol2}
        def policy_mapping_fn(aid, **kwargs):
            return "pol1" if aid in {"player_0", "player_1"} else "pol2"
    else:
        policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
        def policy_mapping_fn(aid, **kwargs):
            return aid
    return env_config, policies, policy_mapping_fn




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

pbt_interval = args.checkpoint_interval if args.pbt else 10_000_000_000_000

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=pbt_interval,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.2),
        "entropy_coeff": lambda: random.uniform(0.01, 0.2),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 10),
        "train_batch_size": [100, 500, 1000],
    },
    custom_explore_fn=explore,
)


if __name__ == "__main__":

    if args.ip:
        ray.init(address=args.ip, _redis_password="longredispassword")

    env_name = "trade_v4"

    register_env(env_name, lambda config: Trade(config))

    env_config, policies, policy_mapping_fn = generate_configs()
    batch_size = args.batch_size

    tune.run(
        ReusablePPOTrainer,
        name=args.name,
        scheduler=pbt,
        metric="episode_reward_mean",
        mode="max",
        resume=False,
        num_samples=1,
        stop={"timesteps_total": args.num_steps},
        checkpoint_freq=args.checkpoint_interval,
        reuse_actors=True,
        local_dir=f"{RESULTS_DIR}/{args.class_name}",
        config={
            # Environment specific
            "env": env_name,
            "env_config": env_config,
            "callbacks": TradeCallback,
            "recreate_failed_workers": True,
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
            'rollout_fragment_length': 50,
            'lr': tune.choice([1e-04, 1e-05]),
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            }})
