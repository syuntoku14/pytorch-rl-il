import unittest
from rlil.environments import GymEnvironment
from rlil.presets.validate_agent import validate_agent
from rlil.presets.continuous import ddpg, sac
# import ptvsd
# ptvsd.enable_attach()
# ptvsd.wait_for_attach()


class TestContinuousPresets(unittest.TestCase):
    def test_ddpg(self):
        self.validate(ddpg(replay_start_size=50, device='cpu'))

    def test_sac(self):
        self.validate(sac(replay_start_size=50, device='cpu'))

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment('Pendulum-v0'))

if __name__ == '__main__':
    unittest.main()
