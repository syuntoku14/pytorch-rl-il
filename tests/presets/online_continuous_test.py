import pytest
from rlil.environments import GymEnvironment
from rlil.presets.continuous import ddpg, sac, td3, ppo
from rlil.presets import validate_agent


def test_ddpg():
    env = GymEnvironment("MountainCarContinuous-v0")
    validate_agent(ddpg(replay_start_size=50), env, done_step=50)


def test_sac():
    env = GymEnvironment("MountainCarContinuous-v0")
    validate_agent(sac(replay_start_size=50), env, done_step=50)


def test_td3():
    env = GymEnvironment("MountainCarContinuous-v0")
    validate_agent(td3(replay_start_size=50), env, done_step=50)


def test_ppo():
    env = GymEnvironment("MountainCarContinuous-v0")
    validate_agent(ppo(replay_start_size=5), env, done_step=50)
