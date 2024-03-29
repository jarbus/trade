import numpy as np
import sys
from trade_v4 import Trade, METABOLISM, PLACE_AMOUNT, POLICY_MAPPING_FN
import unittest
import random

"""
Add tests for:
    move
    light
"""
p0 = "player_0"
p1 = "player_1"
default_config = {
        "window": (1, 1),
        "grid": (3, 3),
        "food_types": 2,
        "num_agents": 2,
        "episode_length": 200,
        "move_coeff": 0.1,
        "dist_coeff": 0.1,
        "twonn_coeff": 0.1,
        "death_prob": 0.1,
        "food_env_spawn": 10.0,
        "random_start": False,
        "respawn": True,
        "survival_bonus": 1,
        "punish": False,
        "punish_coeff": 2,
        "policy_mapping_fn": POLICY_MAPPING_FN[1],
        "vocab_size": 0}

# def act(env, *acts):
#     return {agent: env.MOVES.index(act) for agent, act in zip(env.agents, acts)}

def act(env, acts):
    return {agent: env.MOVES.index(act) for agent, act in acts.items()}

class TestTrade(unittest.TestCase):

    def test_exchange(self):
        config = default_config.copy()
        config["grid"] = (11, 11)
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        counts = env.agent_food_counts[p0], env.agent_food_counts[p1]
        env.step(act(env, {"player_0": "PLACE_0"}))
        env.step(act(env, {"player_1": "PLACE_1"}))
        env.step(act(env, {"player_0": "PICK_1"}))
        env.step(act(env, {"player_1": "PICK_0"}))
        self.assertAlmostEqual(counts[0][0]-2*METABOLISM, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(counts[1][1]-2*METABOLISM, env.agent_food_counts[p1][1])
        self.assertAlmostEqual(counts[0][1]-2*METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p0][1])
        self.assertAlmostEqual(counts[1][0]-2*METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p1][0])
        self.assertAlmostEqual(env.mc.num_exchanges[0], PLACE_AMOUNT)
        self.assertAlmostEqual(env.mc.num_exchanges[1], PLACE_AMOUNT)


    # Legacy test
    # def test_punish(self):
    #     """Test a step with no punish, and a step with a punish,
    #     and confirm the reward is the same"""
    #     no_pun_env = Trade(default_config)
    #     no_pun_env.reset()
    #     no_pun_env.agent_positions[p0] = (0,0)
    #     no_pun_env.agent_positions[p1] = (0,0)
    #     no_pun_obs, no_pun_rew, _, _ = no_pun_env.step(act(no_pun_env, "PICK_1", "PICK_0"))


    #     pun_oned_config = default_config.copy()
    #     pun_oned_config["punish"] = True
    #     pun_env = Trade(pun_oned_config)
    #     pun_env.reset()
    #     pun_env.agent_positions[p0] = (0,0)
    #     pun_env.agent_positions[p1] = (0,0)
    #     pun_obs, pun_rew, _, _ = pun_env.step(act(pun_env, "PUNISH", "PUNISH"))
    #     self.assertAlmostEqual(pun_rew[p0] + pun_oned_config["punish_coeff"], no_pun_rew[p0])
    #     self.assertAlmostEqual(pun_rew[p1] + pun_oned_config["punish_coeff"], no_pun_rew[p1])

    def test_pick(self):
        config = default_config.copy()
        config["grid"] = (11,11)
        config["food_env_spawn"] = 10.0
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        print(env.table.sum(axis=3).sum(axis=2).round(2))
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 2] = 1
        env.table[0, 0, 1, 2] = 1
        fc = env.agent_food_counts[p0].copy(), env.agent_food_counts[p1].copy()
        env.step(act(env, {"player_0": "PICK_0"}))
        env.step(act(env, {"player_1": "PICK_1"}))
        self.assertAlmostEqual(fc[0][0] - METABOLISM + 1, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(fc[1][1] - METABOLISM + 1, env.agent_food_counts[p1][1])


    def test_place(self):
        pass

    def test_light(self):
        # Create config with 7x7 grid
        config = default_config.copy()
        config["grid"] = (11, 11)
        center = tuple(c // 2 for c in config["grid"])
        env = Trade(config)
        env.reset()
        # Check that light.contains returns True for day time
        env.light.light_level = 0.5
        env.light.frame = env.light.fire_frame()
        self.assertTrue(env.light.contains((0, 0)))
        # Check that light.contains returns False for night time
        env.light.light_level = -1
        env.light.frame = env.light.fire_frame()
        self.assertFalse(env.light.contains((0, 0)))
        # Check that light.contains returns True for night time near center
        self.assertTrue(env.light.contains(center))





    def test_policy_frames(self):
        # Confirm that the sum of policy frames across different policy splits is the same
        pol_conf = default_config.copy()
        pol_conf["num_agents"] = 4
        pol_conf["window"] = (1,1)
        pol_conf["grid"] = (11,11)
        pols = [1,2,4]
        obss = {p: {} for p in pols}
        for num_policies in pols:
            pol_conf["policy_mapping_fn"] = POLICY_MAPPING_FN[num_policies]
            env = Trade(pol_conf)
            env.reset()
            env.agent_positions["player_0"] = (0,0)
            env.agent_positions["player_1"] = (0,0)
            env.agent_positions["player_2"] = (0,0)
            env.agent_positions["player_3"] = (0,0)
            env.table = np.zeros(env.table.shape)
            for agent in env.agents:
                obs, _, _, _ = env.step(act(env, {agent: "NONE"}))
                obss[num_policies].update(obs)
        frames_per_policy = 3
        # confirm observations have expected shapes
        self.assertEquals(obss[2]["player_0"].shape[0], obss[1]["player_0"].shape[0] + frames_per_policy)
        self.assertEquals(obss[4]["player_0"].shape[0], obss[1]["player_0"].shape[0] + 3*frames_per_policy)

        # confirm self frames are all equal
        def pol_slice(i):
            return slice(2+i*frames_per_policy, 2+(i+1)*frames_per_policy)
        self.assertTrue(np.array_equal(obss[1]["player_0"][pol_slice(1)], obss[2]["player_0"][pol_slice(2)]))
        self.assertTrue(np.array_equal(obss[1]["player_0"][pol_slice(1)], obss[4]["player_0"][pol_slice(4)]))

        def other_pols(i, n):
            return slice(2+i*frames_per_policy, 2+(i+n)*frames_per_policy)
        obs1 = np.where(obss[1]["player_0"][other_pols(0, 1)] < 0, 0, obss[1]["player_0"][other_pols(0, 1)]).sum(axis=0)
        obs2 = np.where(obss[2]["player_0"][other_pols(0, 2)] < 0, 0, obss[2]["player_0"][other_pols(0, 2)]).sum(axis=0)
        obs4 = np.where(obss[4]["player_0"][other_pols(0, 4)] < 0, 0, obss[4]["player_0"][other_pols(0, 4)]).sum(axis=0)
        self.assertTrue(np.array_equal(obs2, obs4))
        self.assertTrue(np.array_equal(obs1, obs2))


    def test_all_functions_run(self):
        env = Trade(default_config)
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

    def test_env_run(self):
        conf = default_config.copy()
        conf["punish"] = True
        conf["num_agents"] = 3
        env = Trade(conf)
        env.reset()

        env.agent_positions["player_0"] = (0,0)
        env.agent_positions["player_1"] = (0,0)
        env.agent_positions["player_2"] = (0,0)

        def random_actions():
            return {agent: random.randint(0, len(env.MOVES)-1) for agent in env.agents if not env.compute_done(agent)}

        obs, rews, dones, _ = env.step(act(env, "PUNISH", "PUNISH"))
        while not dones["__all__"]:
            obs, rews, dones, _ = env.step(random_actions())
        # Check rewards
        self.assertNotAlmostEqual(env.mc.rew_health, 0)
        self.assertNotAlmostEqual(env.mc.rew_nn,     0)
        self.assertNotAlmostEqual(env.mc.rew_twonn,  0)
        self.assertNotAlmostEqual(env.mc.rew_pun,    0)
        self.assertNotAlmostEqual(env.mc.rew_mov,    0)
        self.assertNotAlmostEqual(env.mc.rew_other_survival_bonus,    0)


import unittest
def testsuite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrade("test_pick"))
    # suite.addTest(TestTrade("test_exchange"))
    # suite.addTest(TestTrade("test_light"))
    #suite.addTest(TestTrade("test_policy_frames"))
    return suite

if __name__ == '__main__':
    # unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(testsuite())
