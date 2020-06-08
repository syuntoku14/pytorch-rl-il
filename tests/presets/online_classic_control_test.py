import pytest
from rlil.environments import GymEnvironment
from rlil.presets.classic_control import dqn
from rlil.presets import env_validation, trainer_validation
from rlil.initializer import set_device


def test_dqn():
    env = GymEnvironment("CartPole-v0", append_time=True)
    env_validation(dqn(replay_start_size=1), env, done_step=50)
    trainer_validation(dqn(replay_start_size=1), env)


def test_n_step():
    env = GymEnvironment("CartPole-v0", append_time=True)
    for preset in [dqn, ]:
        agent_fn = preset(n_step=5)
        agent = agent_fn(env)
        lazy_agent = agent.make_lazy_agent()
        lazy_agent.set_replay_buffer(env)
        assert lazy_agent._n_step == 5


def test_prioritized():
    env = GymEnvironment("CartPole-v0", append_time=True)
    for preset in [dqn, ]:
        env_validation(preset(prioritized=True, replay_start_size=1),
                       env, done_step=50)


@pytest.mark.skip
def test_apex():
    env = GymEnvironment("CartPole-v0", append_time=True)
    for preset in [dqn, ]:
        trainer_validation(
            preset(replay_start_size=1, use_apex=True), env, apex=True)
