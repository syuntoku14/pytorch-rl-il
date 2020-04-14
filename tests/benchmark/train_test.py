import pytest
import torch
import numpy as np
from rlil.environments import GymEnvironment, State
from rlil.presets.continuous import ddpg, sac, td3
from rlil.presets import validate_agent


def collect_samples(agent, env):
    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))


def test_ddpg(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent_fn = ddpg(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent._should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_sac(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent_fn = sac(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent._should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_td3(benchmark):
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent_fn = td3(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent._should_train()
    benchmark.pedantic(agent.train, rounds=100)
