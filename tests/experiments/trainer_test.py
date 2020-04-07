import unittest
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
from ..mock_agent import MockAgent


class TestTrainer(unittest.TestCase):
    def setUp(self):
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

    def test_trainer_frames(self):
        max_frames = 100
        trainer = Trainer(self.agent, self.sampler, max_frames)
        trainer.start_training()
        assert trainer._writer.frames > max_frames

    def test_trainer_episodes(self):
        max_episodes = 5
        trainer = Trainer(self.agent, self.sampler, max_episodes=max_episodes)
        trainer.start_training()
        assert trainer._writer.frames > max_episodes
