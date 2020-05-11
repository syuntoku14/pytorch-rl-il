import pytest
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ray
from rlil.environments import GymEnvironment
from rlil import nn
from rlil.experiments import Trainer
from rlil.samplers import AsyncSampler
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import set_replay_buffer
from rlil.presets.continuous import sac
from ..mock_agent import MockAgent


@pytest.fixture
def setUp():
    ray.init(include_webui=False, ignore_reinit_error=True)

    env = GymEnvironment('LunarLanderContinuous-v2')

    replay_buffer_size = 100000
    replay_buffer = ExperienceReplayBuffer(replay_buffer_size, env)
    set_replay_buffer(replay_buffer)

    agent = MockAgent(env)
    num_workers = 3
    sampler = AsyncSampler(
        env,
        num_workers=num_workers,
    )

    yield env, agent, sampler


def test_trainer_frames(setUp):
    max_sample_frames = 100
    env, agent, sampler = setUp
    trainer = Trainer(agent, sampler, max_sample_frames=max_sample_frames)
    trainer.start_training()
    assert trainer._writer.sample_frames > max_sample_frames


def test_trainer_episodes(setUp):
    max_sample_episodes = 5
    env, agent, sampler = setUp
    trainer = Trainer(agent, sampler, max_sample_episodes=max_sample_episodes)
    trainer.start_training()
    assert trainer._writer.sample_frames > max_sample_episodes


def test_training(setUp):
    env, agent, sampler = setUp
    agent_fn = sac(replay_start_size=50)
    agent = agent_fn(env)

    trainer = Trainer(agent, sampler, max_sample_episodes=5)
    trainer.start_training()
