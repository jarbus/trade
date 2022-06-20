import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v4 import Trade
from tune import generate_configs
from args import get_args
import os
import torch

args = get_args()
N = 20
if __name__ == "__main__":
    path = f"/home/garbus/trade/serves"
    for name in ["CLASS_NAME", "EXP_NAME", "TRIAL_NAME", "CHECK_NAME"]:
        path = os.path.join(path, name)
        if not os.path.exists(path):
            os.mkdir(path)
    for i in range(0, N):
        f = open(os.path.join(path, f"{i}.out"), "w")

        env_name = "trade_v4"

        register_env(env_name, lambda config: Trade(config))

        num_agents = 4


        env_config, policies, policy_mapping_fn = generate_configs()
        test_env = Trade(env_config)

        trainer = ppo.PPOTrainer(
            config={
                # Environment specific
                "env": env_name,
                "env_config": env_config,
                # General
                "framework": "torch",
                "num_gpus": 1,
                "explore": False,
                "sgd_minibatch_size": 64,
                "num_workers": 1,
                # Method specific
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": policy_mapping_fn,
                },

            },
        )

       # trainer.restore("/work/garbus/ray_results/2d4a/random_food_pos,pbt=100/ReusablePPOTrainer_trade_v3_3e62f_00000_0_lr=0.001_2022-06-01_10-00-52/checkpoint_003500/checkpoint-3500")

        trainer.restore("CHECKPOINT_PATH")
        obss = test_env.reset()
        exchanges = test_env.player_exchanges.copy()
        states = {}
        for agent in obss.keys():
            policy = trainer.get_policy("pol1")
            states[agent] = policy.get_initial_state()

        actions = {agent: None for agent in obss.keys()}
        for i in range(100):
            f.write(f"--------STEP-{i}--------\n")
            test_env.render(out=f)
            for agent in obss.keys():
                policy = trainer.get_policy("pol1")
                actions[agent], states[agent], logits = policy.compute_single_action(obs=np.array(obss[agent]), state=states[agent], policy_id=agent)
                actions[agent] = torch.tensor(actions[agent])

            obss, rews, dones, infos = test_env.step({agent: action for agent, action in actions.items() if not test_env.compute_done(agent)})
            for key in test_env.player_exchanges:
                if test_env.player_exchanges[key] > exchanges[key]:
                    giver, taker, food_type = key
                    amount = test_env.player_exchanges[key] - exchanges[key]
                    f.write(f"Exchange: {giver} gave {amount} of food {food_type} to {taker}\n")
            exchanges = test_env.player_exchanges.copy()
            if dones["__all__"]:
                f.write("--------FINAL-STEP--------\n")
                test_env.render(out=f)
                f.write("game over\n")
                break
        f.close()
