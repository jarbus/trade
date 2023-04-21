import numpy as np
import sys
from trade_v4 import Trade, METABOLISM, PLACE_AMOUNT, POLICY_MAPPING_FN
import unittest
import random
def act(env, acts):
    return {agent: env.MOVES.index(act) for agent, act in acts.items()}

def random_actions(env):
    return {agent: random.randint(0, len(env.MOVES)-1) for agent in env.agents if not env.compute_done(agent)}

p0 = "f0a0"
p1 = "f1a0"
default_config = {
        "window": (1, 1),
        "grid": (11, 11),
        "food_types": 2,
        "num_agents": 2,
        "episode_length": 200,
        #"move_coeff": 0.1,
        #"dist_coeff": 0.1,
        "fires": [(5,5)],
        "foods": [(0, 0,0), (1,0,0)],
        "matchups": [(p0, p1)],
        #"twonn_coeff": 0.1,
        #"death_prob": 0.1,
        "food_env_spawn": 10.0,
        "random_start": False,
        "caps": [1, 1],
        "respawn": True,
        "survival_bonus": 1,
        "punish": False,
        "punish_coeff": 2,
        "policy_mapping_fn": POLICY_MAPPING_FN,
        "vocab_size": 0}





class TestTrade(unittest.TestCase):

    def test_no_act_drop(self):
        """Test that agents don't get action reward from dropped food"""
        config = default_config.copy()
        config["caps"] = (1 ,  1)
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 0] = 1
        env.table[0, 0, 1, 1] = 1
        self.assertEqual(env.action_rewards[p0], 0)
        pick_amt = env.compute_pick_amount(0,0,0,"f0a0")
        self.assertEqual(pick_amt, 0)
        env.step(act(env, {"f0a0": "PICK_0"}))
        self.assertEqual(env.action_rewards[p0], 0)

    def test_cap_spawn(self):
        """Test that agents can't pick up more food than
        their caps for spawned food"""
        config = default_config.copy()
        config["caps"] = (1 ,  1)
        config["grid"] = (11, 11)
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 2] = 2
        env.table[0, 0, 1, 2] = 2
        env.table[0, 0, 0, 0] = 2 # this should not get picked
        env.step(act(env, {"f0a0": "PICK_0"}))
        env.step(act(env, {"f1a0": "PICK_1"}))

        self.assertEqual(env.agent_food_counts[p0][0],
                         env.caps[0] - METABOLISM)
        self.assertEqual(env.agent_food_counts[p1][1],
                         env.caps[1] - METABOLISM)
        self.assertEqual(env.table[0, 0, 0, 2], 1)
        self.assertEqual(env.table[0, 0, 1, 2], 1)
        self.assertEqual(env.table[0, 0, 0, 0], 2)
        self.assertEqual(env.action_rewards[p0], 1)
        self.assertEqual(env.action_rewards[p1], 1)

    def test_cap_drop(self):
        """Test that agents can't pick up more food than
        their caps for dropped food"""
        config = default_config.copy()
        config["caps"] = (1 ,  1)
        config["grid"] = (11, 11)
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 0] = 0.5
        env.table[0, 0, 0, 1] = 1.5
        env.table[0, 0, 1, 0] = 1.5
        env.table[0, 0, 1, 1] = 0.5
        env.step(act(env, {"f0a0": "PICK_0"}))
        env.step(act(env, {"f1a0": "PICK_1"}))

        self.assertEqual(env.agent_food_counts[p0][0],
                         env.caps[0] - METABOLISM)
        self.assertEqual(env.agent_food_counts[p1][1],
                         env.caps[1] - METABOLISM)

        self.assertEqual(env.action_rewards[p0], 1)
        self.assertEqual(env.action_rewards[p1], 1)
        self.assertEqual(env.table[0, 0, 0, 0], 0)
        self.assertEqual(env.table[0, 0, 0, 1], 1)
        self.assertEqual(env.table[0, 0, 1, 0], 0.5)
        self.assertEqual(env.table[0, 0, 1, 1], 0.5)

    def test_deterministic_spawn(self):
        config = default_config.copy()
        config["grid"] = (11, 11)
        env = Trade(config)
        env.reset()
        # Check that the agents are spawned in the same place after reset
        agent_pos = env.agent_positions.copy()
        env.reset()
        self.assertEqual(agent_pos, env.agent_positions)
        env = Trade(config)
        env.reset()
        self.assertEqual(agent_pos, env.agent_positions)

    def test_agent_spawner(self):
        config = default_config.copy()
        config["grid"] = (11, 11)
        env = Trade(config)
        for _ in range(2):
            env.reset()
            centers = [(fc[1], fc[2]) for fc in env.foods]
            for agent, center in zip(env.agents, centers):
                self.assertEqual(env.agent_positions[agent], center)

    def test_exchange(self):
        config = default_config.copy()
        config["grid"] = (11, 11)
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.agent_food_counts[p0] = [2,0]
        env.agent_food_counts[p1] = [0,2]
        env.table = np.zeros(env.table.shape)
        counts = env.agent_food_counts[p0], env.agent_food_counts[p1]
        env.step(act(env, {"player_0": "PLACE_0"}))
        env.step(act(env, {"player_1": "PLACE_1"}))
        env.step(act(env, {"player_0": "PICK_1"}))
        env.step(act(env, {"player_1": "PICK_0"}))
        self.assertAlmostEqual(counts[0][0]-2*METABOLISM, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(counts[1][1]-2*METABOLISM, env.agent_food_counts[p1][1])
        # we subtract a single unit of metabolism because we only take a single step with this resource
        self.assertAlmostEqual(counts[0][1]-METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p0][1])
        self.assertAlmostEqual(counts[1][0]-METABOLISM+PLACE_AMOUNT, env.agent_food_counts[p1][0])
        self.assertAlmostEqual(env.mc.num_exchanges[0], PLACE_AMOUNT)
        self.assertAlmostEqual(env.mc.num_exchanges[1], PLACE_AMOUNT)

    def test_pick_specialty(self):
        config = default_config.copy()
        config["grid"] = (11,11)
        config["food_env_spawn"] = 10.0
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 2] = 1
        env.table[0, 0, 1, 2] = 1
        fc = env.agent_food_counts[p0].copy(), env.agent_food_counts[p1].copy()
        env.step(act(env, {"f0a0": "PICK_0"}))
        env.step(act(env, {"f1a0": "PICK_1"}))
        self.assertAlmostEqual(fc[0][0] - METABOLISM + 1, env.agent_food_counts[p0][0])
        self.assertAlmostEqual(fc[1][1] - METABOLISM + 1, env.agent_food_counts[p1][1])
        self.assertAlmostEqual(env.mc.picked_counts["f0a0"][0], 1)
        self.assertAlmostEqual(env.mc.picked_counts["f1a0"][1], 1)

    def test_pick_nonspecialty(self):
        config = default_config.copy()
        config["grid"] = (11,11)
        config["food_env_spawn"] = 10.0
        env = Trade(config)
        env.reset()
        env.agent_positions[p0] = (0,0)
        env.agent_positions[p1] = (0,0)
        env.table = np.zeros(env.table.shape)
        env.table[0, 0, 0, 2] = 1
        env.table[0, 0, 1, 2] = 1
        fc = env.agent_food_counts[p0].copy(), env.agent_food_counts[p1].copy()
        env.step(act(env, {"f0a0": "PICK_1"}))
        env.step(act(env, {"f1a0": "PICK_0"}))
        self.assertAlmostEqual(fc[0][1] - METABOLISM + 0.5, env.agent_food_counts[p0][1])
        self.assertAlmostEqual(fc[1][0] - METABOLISM + 0.5, env.agent_food_counts[p1][0])
        self.assertAlmostEqual(env.mc.picked_counts["f0a0"][1], 0.5)
        self.assertAlmostEqual(env.mc.picked_counts["f1a0"][0], 0.5)

    def test_light(self):
        # Create config with 7x7 grid
        config = default_config.copy()
        config["grid"] = (11, 11)
        config["fires"] = [(5, 5)]
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


    def test_random_actions(self):
        conf = default_config.copy()
        conf["punish"] = False
        conf["num_agents"] = 3
        env = Trade(conf)
        env.reset()
    
        env.agent_positions["player_0"] = (0,0)
        env.agent_positions["player_1"] = (0,0)
        env.agent_positions["player_2"] = (0,0)
    
        dones = {"__all__": False}
        while not dones["__all__"]:
            obs, rews, dones, infos = env.step(random_actions(env))
            for agent, values in obs.items():
                self.assertEqual(env.obs_size, values.shape)

    def test_render(self):
        conf = default_config.copy()
        conf["punish"] = False
        conf["food_agent_start"] = 1
        conf["num_agents"] = 3
        conf["render_str"] = "/tmp/test.out"
        env = Trade(conf)
        env.reset()
    
        env.agent_positions["player_0"] = (0,0)
        env.agent_positions["player_1"] = (0,0)
        env.agent_positions["player_2"] = (0,0)
    
        dones = {"__all__": False}
        while not dones["__all__"]:
            obs, rews, dones, infos = env.step(random_actions(env))
            for agent, values in obs.items():
                self.assertEqual(env.obs_size, values.shape)

    #def test_policy_frames(self):
    #    # Confirm that the sum of policy frames across different policy splits is the same
    #    pol_conf = default_config.copy()
    #    pol_conf["num_agents"] = 4
    #    pol_conf["window"] = (1,1)
    #    pol_conf["grid"] = (11,11)
    #    pol_conf["matchups"] = [("player_0", "player_1", "player_2", "player_3")] 
    #    pols = [1,2,4]
    #    obss = {p: {} for p in pols}
    #    for num_policies in pols:
    #        pol_conf["policy_mapping_fn"] = POLICY_MAPPING_FN[num_policies]
    #        env = Trade(pol_conf)
    #        env.reset()
    #        env.agent_positions["player_0"] = (0,0)
    #        env.agent_positions["player_1"] = (0,0)
    #        env.agent_positions["player_2"] = (0,0)
    #        env.agent_positions["player_3"] = (0,0)
    #        env.table = np.zeros(env.table.shape)
    #        for agent in env.agents:
    #            obs, _, _, _ = env.step(act(env, {agent: "NONE"}))
    #            obss[num_policies].update(obs)
    #    frames_per_policy = 3
    #    # confirm observations have expected shapes
    #    self.assertEquals(obss[2]["player_0"].shape[0], obss[1]["player_0"].shape[0] + frames_per_policy)
    #    self.assertEquals(obss[4]["player_0"].shape[0], obss[1]["player_0"].shape[0] + 3*frames_per_policy)

    #    # confirm self frames are all equal
    #    def pol_slice(i):
    #        return slice(2+i*frames_per_policy, 2+(i+1)*frames_per_policy)
    #    self.assertTrue(np.array_equal(obss[1]["player_0"][pol_slice(1)], obss[2]["player_0"][pol_slice(2)]))
    #    self.assertTrue(np.array_equal(obss[1]["player_0"][pol_slice(1)], obss[4]["player_0"][pol_slice(4)]))

    #    def other_pols(i, n):
    #        return slice(2+i*frames_per_policy, 2+(i+n)*frames_per_policy)
    #    obs1 = np.where(obss[1]["player_0"][other_pols(0, 1)] < 0, 0, obss[1]["player_0"][other_pols(0, 1)]).sum(axis=0)
    #    obs2 = np.where(obss[2]["player_0"][other_pols(0, 2)] < 0, 0, obss[2]["player_0"][other_pols(0, 2)]).sum(axis=0)
    #    obs4 = np.where(obss[4]["player_0"][other_pols(0, 4)] < 0, 0, obss[4]["player_0"][other_pols(0, 4)]).sum(axis=0)
    #    self.assertTrue(np.array_equal(obs2, obs4))
    #    self.assertTrue(np.array_equal(obs1, obs2))


import unittest
def testsuite():
    suite = unittest.TestSuite()
    # suite.addTest(TestTrade("test_pick_specialty"))
    # suite.addTest(TestTrade("test_pick_nonspecialty"))
    # suite.addTest(TestTrade("test_cap_spawn"))
    # suite.addTest(TestTrade("test_cap_drop"))
    suite.addTest(TestTrade("test_no_act_drop"))
    # suite.addTest(TestTrade("test_exchange"))
    # suite.addTest(TestTrade("test_light"))
    #suite.addTest(TestTrade("test_policy_frames"))
    return suite

if __name__ == '__main__':
    # unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(testsuite())
