import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v3 import Trade

if __name__ == "__main__":

    env_name = "trade_v3"

    register_env(env_name, lambda config: Trade(config))

    num_agents = 2


    env_config = {"window": (2, 2),
                  "grid": (1, 7),
                  "empathy": 1,
                  "food_types": 2,
                  "num_agents": 2,
                  "episode_length": 100,
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

    trainer.restore("/work/garbus/ray_results/reward_sweep/reward sweep 10m vocab=0 grid=(1, 7)/ReusablePPOTrainer_trade_v3_3da8e_00000_0_clip_param=0.2,entropy_coeff=0.05,death_prob=0.05,dist_coeff=0.2,move_coeff=0.2,num_sgd__2022-05-27_12-01-18/checkpoint_004900/checkpoint-4900")

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
