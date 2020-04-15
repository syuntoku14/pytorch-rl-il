import pytest
from rlil.environments import GymEnvironment
from rlil.presets.online.continuous import ddpg, sac, td3
from rlil.presets import validate_agent


def test_ddpg():
    env = GymEnvironment('LunarLanderContinuous-v2')
    validate_agent(ddpg(replay_start_size=50), env)


def test_sac():
    env = GymEnvironment('LunarLanderContinuous-v2')
    validate_agent(sac(replay_start_size=50), env)


def test_td3():
    env = GymEnvironment('LunarLanderContinuous-v2')
    validate_agent(td3(replay_start_size=50), env)
