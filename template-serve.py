import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v3 import Trade
from tune import generate_configs
import os

N = 5
if __name__ == "__main__":
    name = "EXP_NAME"
    path = f"/home/garbus/trade/serves/{name}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, "CHECK_NAME")
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(0, N):
        f = open(os.path.join(path, f"{i}.out"), "w")

        env_name = "trade_v3"

        register_env(env_name, lambda config: Trade(config))

        num_agents = 4


        env_config, policies = generate_configs()
        test_env = Trade(env_config)

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

        trainer.restore("CHECKPOINT_PATH")
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
