import pytest
from rlil.environments import GymEnvironment
from rlil.presets.continuous import ddpg, sac, td3, noisy_td3, ppo, rs_mpc
from rlil.presets import env_validation, trainer_validation


def test_ddpg():
    env = GymEnvironment("MountainCarContinuous-v0")
    env_validation(ddpg(replay_start_size=50), env, done_step=50)
    trainer_validation(ddpg(replay_start_size=50), env)


def test_sac():
    env = GymEnvironment("MountainCarContinuous-v0")
    env_validation(sac(replay_start_size=50), env, done_step=50)
    trainer_validation(sac(replay_start_size=50), env)


def test_td3():
    env = GymEnvironment("MountainCarContinuous-v0")
    env_validation(td3(replay_start_size=50), env, done_step=50)
    trainer_validation(td3(replay_start_size=50), env)


def test_noisy_td3():
    env = GymEnvironment("MountainCarContinuous-v0")
    env_validation(noisy_td3(replay_start_size=50), env, done_step=50)
    trainer_validation(noisy_td3(replay_start_size=50), env)


def test_ppo():
    env = GymEnvironment("MountainCarContinuous-v0")
    env_validation(ppo(replay_start_size=5), env, done_step=50)
    trainer_validation(ppo(replay_start_size=50), env)


def test_rs_mpc():
    env = GymEnvironment("Pendulum-v0")
    env_validation(rs_mpc(replay_start_size=5), env, done_step=50)
    trainer_validation(rs_mpc(replay_start_size=5), env)
