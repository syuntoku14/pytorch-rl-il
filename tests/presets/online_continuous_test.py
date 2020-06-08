import pytest
from rlil.environments import GymEnvironment
from rlil.presets.continuous import vac, ddpg, sac, td3, noisy_td3, ppo, rs_mpc
from rlil.presets import env_validation, trainer_validation
from rlil.initializer import set_device


def test_vac():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(vac(replay_start_size=50), env, done_step=50)
    trainer_validation(vac(replay_start_size=50), env)


def test_ddpg():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(ddpg(replay_start_size=50), env, done_step=50)
    trainer_validation(ddpg(replay_start_size=50), env)


def test_sac(use_cpu):
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(sac(replay_start_size=50), env, done_step=50)
    trainer_validation(sac(replay_start_size=50), env)


def test_n_step():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    for preset in [ddpg, td3, sac]:
        agent_fn = preset(n_step=5)
        agent = agent_fn(env)
        lazy_agent = agent.make_lazy_agent()
        lazy_agent.set_replay_buffer(env)
        assert lazy_agent._n_step == 5


def test_prioritized():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    for preset in [ddpg, td3, sac]:
        env_validation(preset(prioritized=True, replay_start_size=50),
                       env, done_step=50)


def test_td3():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(td3(replay_start_size=50), env, done_step=50)
    trainer_validation(td3(replay_start_size=50), env)


def test_noisy_td3():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(noisy_td3(replay_start_size=50), env, done_step=50)
    trainer_validation(noisy_td3(replay_start_size=50), env)


def test_ppo():
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    env_validation(ppo(replay_start_size=5), env, done_step=50)
    trainer_validation(ppo(replay_start_size=50), env)


def test_rs_mpc():
    env = GymEnvironment("Pendulum-v0", append_time=True)
    env_validation(rs_mpc(replay_start_size=5), env, done_step=50)
    trainer_validation(rs_mpc(replay_start_size=5), env)


def test_apex(use_cpu):
    env = GymEnvironment("LunarLanderContinuous-v2", append_time=True)
    for preset in [ddpg, td3, sac]:
        trainer_validation(
            preset(replay_start_size=5, use_apex=True), env, apex=True)
