import numpy as np
from trade_v4 import Trade, METABOLISM, PLACE_AMOUNT
import unittest

"""
Add tests for:
    move
    light
"""
p0 = "player_0"
p1 = "player_1"
oned_config = {
        "window": (1, 1),
        "grid": (2, 2),
        "food_types": 2,
        "num_agents": 2,
        "episode_length": 200,
        "move_coeff": 0.0,
        "dist_coeff": 0,
        "death_prob": 0.1,
        "random_start": False,
        "respawn": False,
        "survival_bonus": 1,
        "punish": False,
        "punish_coeff": 2,
        "vocab_size": 0}

def act(env, *acts):
    return {agent: env.MOVES.index(act) for agent, act in zip(env.agents, acts)}

class TestTrade(unittest.TestCase):

    def test_exchange(self):
        env = Trade(oned_config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        counts = env.agent_food_counts[p0], env.agent_food_counts[p1]
        env.step(act(env, "PLACE_0", "PLACE_1"))
        env.step(act(env, "PICK_1", "PICK_0"))
        self.assertAlmostEqual(counts[0][0]-2*METABOLISM, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(counts[1][1]-2*METABOLISM, env.agent_food_counts[p1][1])
        self.assertAlmostEqual(counts[0][1]-2*METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p0][1])
        self.assertAlmostEqual(counts[1][0]-2*METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p1][0])
        self.assertAlmostEqual(env.num_exchanges[0], PLACE_AMOUNT)
        self.assertAlmostEqual(env.num_exchanges[1], PLACE_AMOUNT)


    def test_punish(self):
        """Test a step with no punish, and a step with a punish,
        and confirm the reward is the same"""
        no_pun_env = Trade(oned_config)
        no_pun_env.reset()
        no_pun_env.agent_positions[p0] = (0,0)
        no_pun_env.agent_positions[p1] = (0,0)
        no_pun_obs, no_pun_rew, _, _ = no_pun_env.step(act(no_pun_env, "PICK_1", "PICK_0"))


        pun_oned_config = oned_config.copy()
        pun_oned_config["punish"] = True
        pun_env = Trade(pun_oned_config)
        pun_env.reset()
        pun_env.agent_positions[p0] = (0,0)
        pun_env.agent_positions[p1] = (0,0)
        pun_obs, pun_rew, _, _ = pun_env.step(act(pun_env, "PUNISH", "PUNISH"))
        self.assertAlmostEqual(pun_rew[p0] + pun_oned_config["punish_coeff"], no_pun_rew[p0])
        self.assertAlmostEqual(pun_rew[p1] + pun_oned_config["punish_coeff"], no_pun_rew[p1])

    def test_pick(self):
        env = Trade(oned_config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 2] = 1
        env.table[0, 0, 1, 2] = 1
        fc = env.agent_food_counts[p0].copy(), env.agent_food_counts[p1].copy()
        env.step(act(env, "PICK_0", "PICK_1"))
        self.assertAlmostEqual(fc[0][0] - METABOLISM + 1, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(fc[1][1] - METABOLISM + 1, env.agent_food_counts[p1][1])


    def test_place(self):
        pass

    def test_light(self):
        pass

    def test_all_functions_run(self):
        env = Trade(oned_config)
        env.reset()
        env.seed()
        env.spawn_food()
        env.render()
        env.compute_observation("player_0")
        env.compute_done("player_0")
        env.compute_reward("player_0")
        env.compute_exchange_amount(0, 0, 0, 0)
        env.compute_pick_amount(0, 0, 0, 0)
        env.update_dones()
        env.step(act(env, "NONE"))

if __name__ == '__main__':
    unittest.main()
