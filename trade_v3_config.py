from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v3 import Trade, TradeCallback

def generate_configs():
    num_agents = 4
# env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 20, "vocab_size": 0}

    env_config = {"window": (3, 3),
                  "grid": (5, 5),
                  "food_types": 2,
                  "num_agents": num_agents,
                  "episode_length": 200,
                  "move_coeff": 0.0,
                  "dist_coeff": 0.1,
                  "death_prob": 0.1,
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
