import pytest
import gym
from rlil.environments import GymEnvironment
from rlil.presets.continuous import bcq, bc
from rlil.presets import env_validation, trainer_validation
from rlil.memory import ExperienceReplayBuffer
from rlil.environments import Action
from rlil.initializer import set_replay_buffer
from copy import deepcopy
from ..mock_agent import MockAgent


def get_transitions(env):
    replay_buffer = ExperienceReplayBuffer(1000, env)
    set_replay_buffer(replay_buffer)
    agent = MockAgent(env)

    while len(agent.replay_buffer) < 200:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))

    return agent.replay_buffer.get_all_transitions(return_cpprb=True)


def test_bcq():
    env = GymEnvironment('LunarLanderContinuous-v2')
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(bcq(transitions), env, done_step=50)
    trainer_validation(bcq(transitions), env, num_workers=0)


def test_bc():
    env = GymEnvironment('LunarLanderContinuous-v2')
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(bc(transitions), env, done_step=50)
    trainer_validation(bc(transitions), env, num_workers=0)
