from ray import tune
from ray.tune.registry import register_env
from ray import shutdown
from ray.rllib.policy.policy import PolicySpec
from tradeenv import Trade, TradeCallback


if __name__ == "__main__":
    shutdown()

    env_name = "trade_v1"

    register_env(env_name, lambda config: Trade(config))

    test_env = Trade({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    num_agents = 2

    def gen_policy(i):
        config = {
            "model": {
                # Change individual keys in that dict by overriding them, e.g.
                "fcnet_hiddens": [64, 64, 64],
                "fcnet_activation": "relu",
                "use_lstm": False,
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    # policy_ids = list(policies.keys())
    policies = {f"player_{a}": PolicySpec(observation_space=obs_space,
                       action_space=act_space,
                       config={"agent_id": a}) for a in range(num_agents)}
    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            "env_config": {
                "food_types": 2,
                "num_agents": num_agents,
                "episode_length": 100,
            },
            "callbacks": TradeCallback,
            # General
            "log_level": "ERROR",
            "framework": "tf",
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',
            # 'use_critic': True,
            'use_gae': True,
            "lambda": 0.9,
            "gamma": .99,
            # "kl_coeff": 0.001,
            # "kl_target": 1000.,
            "clip_param": 0.4,
            'grad_clip': None,
            "entropy_coeff": 0.1,
            'vf_loss_coeff': 0.25,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 10,  # epoc
            'rollout_fragment_length': 128,
            "train_batch_size": 512,
            'lr': 2e-05,
            "clip_actions": True,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda aid, **kwargs: aid),
            }})
