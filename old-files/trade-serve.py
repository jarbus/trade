import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from tradeenv import Trade


if __name__ == "__main__":

    env_name = "trade_v1"

    register_env(env_name, lambda config: Trade(config))

    num_agents = 3
    env_config = {"food_types": num_agents, "num_agents": num_agents, "episode_length": 100}

    test_env = Trade(env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "agent_id": i,
            "model": {
                # Change individual keys in that dict by overriding them, e.g.
                "fcnet_hiddens": [64, 64, 64],
                "fcnet_activation": "relu",
                "use_lstm": True,
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
            "framework": "tf",
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
    trainer.restore("/home/jack/ray_results/trade_v1/PPO/PPO_trade_v1_fcf97_00000_0_2022-04-28_14-06-56/checkpoint_000150/checkpoint-150")

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
