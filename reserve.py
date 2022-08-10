import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from trade_v4 import Trade
from tune import generate_configs
from args import get_args
import os
import torch
from pdb import set_trace

args = get_args()
N = 5
if __name__ == "__main__":
    assert args.checkpoint # name contains information about experiment
    assert args.tmp_checkpoint # copy to tmp in case it gets overwritten or moved by ray

    results_path = "/work/garbus/ray_results/"
    serve_path = f"/home/garbus/trade/serves"
    class_name, exp_name, trial_name, check_name = args.checkpoint[len(results_path):].split("/")[:-1]

    # create paths 
    path = serve_path
    for name in [class_name, exp_name, trial_name, check_name]:
        path = os.path.join(path, name)
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except FileExistsError:
                print(f"Error making dir {path}")
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

        trainer.restore(args.tmp_checkpoint)
        obss = test_env.reset()
        exchanges = test_env.mc.player_exchanges.copy()
        states = {}
        for agent in test_env.agents:
            policy = trainer.get_policy(policy_mapping_fn(agent))
            states[agent] = policy.get_initial_state()

        prev_step = test_env.steps
        while test_env.steps <= test_env.max_steps:
            actions = {}
            if prev_step != test_env.steps:
                f.write(f"--------STEP-{test_env.steps}--------\n")
                test_env.render(out=f)
                prev_step = test_env.steps
            for agent in obss.keys():
                policy = trainer.get_policy(policy_mapping_fn(agent))
                actions[agent], states[agent], logits = policy.compute_single_action(obs=np.array(obss[agent]), state=states[agent], policy_id=agent, explore=False, timestep=i)
                actions[agent] = torch.tensor(actions[agent])

            obss, rews, dones, infos = test_env.step({agent: action for agent, action in actions.items() if not test_env.compute_done(agent)})
            for key in test_env.mc.player_exchanges:
                if test_env.mc.player_exchanges[key] > exchanges[key]:
                    giver, taker, food_type = key
                    amount = test_env.mc.player_exchanges[key] - exchanges[key]
                    f.write(f"Exchange: {giver} gave {amount} of food {food_type} to {taker}\n")
            exchanges = test_env.mc.player_exchanges.copy()
            if dones["__all__"]:
                break
                #f.write("--------FINAL-STEP--------\n")
                #test_env.render(out=f)
                #f.write("game over\n")
            #print("END OF LOOP")
        f.close()
