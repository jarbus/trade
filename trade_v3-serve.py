import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v3 import Trade
import os

N = 5
if __name__ == "__main__":
    name = "experiment-root"
    path = f"/home/garbus/trade/serves/{name}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, "checkpoint_000010")
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(0, N):
        f = open(os.path.join(path, f"{i}.out"), "w")

        env_name = "trade_v3"

        register_env(env_name, lambda config: Trade(config))

        num_agents = 4


        env_config = {"window": (3, 3),
                      "grid": (5, 5),
                      "empathy": 1,
                      "food_types": 2,
                      "num_agents": num_agents,
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

       # trainer.restore("/work/garbus/ray_results/2d4a/random_food_pos,pbt=100/ReusablePPOTrainer_trade_v3_3e62f_00000_0_lr=0.001_2022-06-01_10-00-52/checkpoint_003500/checkpoint-3500")

        trainer.restore("/work/garbus/ray_results/experiment-root/experiment-root/ReusablePPOTrainer_trade_v3_a1f8e_00000_0_lr=0.001_2022-06-04_10-00-06/checkpoint_000010/checkpoint-10")
        obss = test_env.reset()
        states = {}
        for agent in obss.keys():
            policy = trainer.get_policy(agent)
            states[agent] = policy.get_initial_state()

        for i in range(100):
            f.write(f"--------STEP-{i}--------\n")
            test_env.render(out=f)
            actions = {}
            for agent in obss.keys():
                policy = trainer.get_policy(agent)
                actions[agent], states[agent], logits = policy.compute_single_action(obs=np.array(obss[agent]), state=states[agent], policy_id=agent)

            obss, rews, dones, infos = test_env.step({agent: action for agent, action in actions.items() if not test_env.compute_done(agent)})
            if dones["__all__"]:
                f.write("--------FINAL-STEP--------\n")
                test_env.render(out=f)
                f.write("game over\n")
                break
        f.close()
