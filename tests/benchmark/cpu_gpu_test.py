import pytest
import torch
import numpy as np
from rlil.environments import GymEnvironment, State
from rlil.presets.continuous import ddpg


def collect_samples(agent, env):
    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))


def test_ddpg_cuda(benchmark, use_gpu):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent_fn = ddpg(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent.should_train()
    benchmark.pedantic(agent.train, rounds=100)


def test_ddpg_cpu(benchmark, use_cpu):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent_fn = ddpg(replay_start_size=100)
    agent = agent_fn(env)
    collect_samples(agent, env)
    assert agent.should_train()
    benchmark.pedantic(agent.train, rounds=100)