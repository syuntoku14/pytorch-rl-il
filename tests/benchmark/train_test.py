import pytest
import torch
import numpy as np
from rlil.environments import GymEnvironment, State
from rlil.presets.continuous import ddpg, sac, td3, bc
from ..presets.offline_continuous_test import get_transitions


def collect_samples(agent, env):
    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))


def test_ddpg(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    agent_fn = ddpg(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent.should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_sac(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    agent_fn = sac(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent.should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_td3(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    agent_fn = td3(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent.should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_bc(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    transitions = get_transitions(env)
    agent_fn = bc(transitions)
    agent = agent_fn(env)
    assert len(transitions["obs"]) > 100
    benchmark.pedantic(agent.train, rounds=100)
