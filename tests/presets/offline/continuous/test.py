import pytest
import gym
from rlil.environments import GymEnvironment
from rlil.presets.offline.continuous import bcq
from rlil.memory import ExperienceReplayBuffer
from rlil.environments import Action
from rlil.initializer import set_replay_buffer
from copy import deepcopy
from ....mock_agent import MockAgent


def gen_replay_buffer(env):
    replay_buffer = ExperienceReplayBuffer(1000)
    set_replay_buffer(replay_buffer)
    agent = MockAgent(env)
 
    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))

    return deepcopy(agent.replay_buffer)


@pytest.mark.skip
def test_bcq():
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = gen_replay_buffer(env)
    assert len(replay_buffer) > 100


def test_bc():
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = gen_replay_buffer(env)
    assert len(replay_buffer) > 100

