import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v2 import Trade


if __name__ == "__main__":

    env_name = "trade_v2"

    register_env(env_name, lambda config: Trade(config))

    num_agents = 3


    env_config = {"window": (2, 2),
                  "grid": (7, 7),
                  "empathy": 1,
                  "food_types": num_agents,
                  "num_agents": num_agents,
                  "episode_length": 100,
                  "scale": None, #1m random or None
                  "vocab_size": 0}

    test_env = Trade(env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            # "agent_id": i,
            "model": {
                # Change individual keys in that dict by overriding them, e.g.
                "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1]],
                #"conv_filters": [[128, [3, 3], 1], [128, [3, 3], 1], [128, [3, 3], 1]],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [64, 64],
                #"post_fcnet_hiddens": [256, 128, 64],
                "post_fcnet_activation": "relu",
                "use_lstm": True,
                #"use_attention": True,
                #"attention_num_transformer_units": 1,
                #"attention_dim": 64,
                #"attention_num_heads": 2,
                #"attention_memory_inference": 100,
                #"attention_memory_training": 50,
                #"vf_share_layers": False,
                #"attention_use_n_prev_actions": 2,
                #"attention_use_n_prev_rewards": 2,
            },
            "gamma": 0.99,
        }
        return PolicySpec(None, obs_space, act_space, config)
 

    policies = {f"player_{a}": gen_policy(a) for a in range(num_agents)}
    policy_ids = list(policies.keys())

    trainer = ppo.PPOTrainer(
        config={
            # Environment specific
            "env": env_name,
            "env_config": env_config,
            # General
            "framework": "torch",
            "num_gpus": 0,
            "explore": False,
            "sgd_minibatch_size": 64,
            "num_workers": 1,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda aid, **kwargs: aid),
            },

        },
    )

    # trainer.restore("/home/jack/ray_results/trade_v1/PPO/PPO_trade_v1_1bff5_00000_0_2022-04-28_11-37-28/checkpoint_000280/checkpoint-280")
    trainer.restore("/work/garbus/ray_results/scaling/random grid 2f_min/ReusablePPOTrainer_trade_v2_eda5b_00002_2_clip_param=0.1,entropy_coeff=0.2,num_sgd_iter=10_2022-05-22_16-08-59/checkpoint_000600/checkpoint-600")

    obss = test_env.reset()
    states = {}
    for agent in obss.keys():
        policy = trainer.get_policy(agent)
        states[agent] = policy.get_initial_state()

    for i in range(100):
        print(f"--------STEP-{i}--------")
        test_env.render()
        actions = {}
        for agent in obss.keys():
            policy = trainer.get_policy(agent)
            actions[agent], states[agent], logits = policy.compute_single_action(obs=np.array(obss[agent]), state=states[agent], policy_id=agent)

        obss, rews, dones, infos = test_env.step({agent: action for agent, action in actions.items() if not test_env.compute_done(agent)})
        if dones["__all__"]:
            print("--------FINAL-STEP--------")
            test_env.render()
            print("game over")
            break
