import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from tradeenv import Trade


if __name__ == "__main__":

    env_name = "trade_v1"

    register_env(env_name, lambda config: Trade(config))

    test_env = Trade({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space

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

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    trainer = ppo.PPOTrainer(
        config={
            # Environment specific
            "env": env_name,
            # General
            "framework": "tf",
            "num_gpus": 0,
            "num_workers": 1,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[0]),
            },
        },
    )
    trainer.restore("/home/jack/ray_results/trade_v1/PPO/PPO_trade_v1_8807a_00000_0_2022-04-25_14-07-13/checkpoint_001320/checkpoint-1320")

    tenv = Trade({})
    obss = tenv.reset()
    for i in range(100):
        print(f"--------STEP-{i}--------")
        tenv.render()
        actions = trainer.compute_actions(obss, policy_id="policy_0")
        obss, rews, dones, infos = tenv.step({agent: action for agent, action in actions.items() if not tenv.compute_done(agent)})
        if dones["__all__"]:
            print("--------FINAL-STEP--------")
            tenv.render()
            print("game over")
            break
