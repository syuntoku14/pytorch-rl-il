import unittest
import numpy as np
import torch
import gym
import time
import warnings
import ray
from rlil import nn
from rlil.environments import GymEnvironment, Action
from rlil.policies.deterministic import DeterministicPolicyNetwork
from rlil.samplers import AsyncSampler
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import set_replay_buffer
from ..mock_agent import MockAgent


class TestSampler(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        ray.init(include_webui=False, ignore_reinit_error=True)

        self.replay_buffer_size = 100
        replay_buffer = ExperienceReplayBuffer(self.replay_buffer_size)
        set_replay_buffer(replay_buffer)
        self.env = GymEnvironment('LunarLanderContinuous-v2')
        self.agent = MockAgent(self.env)
        self.num_workers = 3
        self.sampler = AsyncSampler(
            self.env,
            num_workers=self.num_workers,
            seed=0,
        )

    def test_sampler_episode(self):
        worker_episodes = 6
        lazy_agent = self.agent.make_lazy_agent()
        self.sampler.start_sampling(
            lazy_agent, worker_episodes=worker_episodes)
        sample_info = self.sampler.store_samples(timeout=1e8)

        assert sample_info["episodes"] == worker_episodes * self.num_workers
        assert len(self.sampler._replay_buffer) == self.replay_buffer_size

    def test_sampler_frames(self):
        worker_frames = 50

        lazy_agent = self.agent.make_lazy_agent()
        self.sampler.start_sampling(
            lazy_agent, worker_frames=worker_frames)
        sample_info = self.sampler.store_samples(timeout=1e8)

        assert sample_info["frames"] > worker_frames * self.num_workers
        assert len(self.sampler._replay_buffer) == self.replay_buffer_size

    def test_ray_wait(self):
        worker_episodes = 100

        lazy_agent = self.agent.make_lazy_agent()
        self.sampler.start_sampling(
            lazy_agent, worker_episodes=worker_episodes)
        self.sampler.store_samples()

        assert len(self.sampler._replay_buffer) == 0
