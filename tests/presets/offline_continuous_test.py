import pytest
import gym
from rlil.environments import GymEnvironment
from rlil.presets.continuous import bcq, bc, vae_bc, bear, brac
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
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(bcq(transitions), env, done_step=50)
    trainer_validation(bcq(transitions), env)


def test_bear():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(bear(transitions), env, done_step=50)
    trainer_validation(bear(transitions), env)


def test_brac():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(brac(transitions, bc_iters=5), env, done_step=50)
    trainer_validation(brac(transitions, bc_iters=5), env)


def test_bc():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(bc(transitions), env, done_step=50)
    trainer_validation(bc(transitions), env)


def test_vae_bc():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    assert len(transitions["obs"]) > 100

    env_validation(vae_bc(transitions), env, done_step=50)
    trainer_validation(vae_bc(transitions), env)
