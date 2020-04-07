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
from rlil.samplers import SyncSampler
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import set_replay_buffer
from ..mock_agent import MockAgent


# class TestSampler(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter("ignore", ResourceWarning)
#         ray.init(include_webui=False, ignore_reinit_error=True)

#         self.replay_buffer_size = 1e6
#         replay_buffer = ExperienceReplayBuffer(self.replay_buffer_size)
#         set_replay_buffer(replay_buffer)
#         self.env = GymEnvironment('LunarLanderContinuous-v2')
#         self.agent = MockAgent(self.env)
#         self.sampler = SyncSampler(
#             self.env,
#             num_workers=3,
#             seed=0,
#         )

#     def test_sampler(self):
#         lazy_agent = self.agent.make_lazy_agent()
#         sum_frames = 0
#         sum_episodes = 0
#         while sum_episodes == 0:
#             self.sampler.start_sampling(lazy_agent)
#             frames, episodes = self.sampler.store_samples()
#             sum_frames += frames
#             sum_episodes += episodes

#         assert len(self.sampler._replay_buffer) == sum_frames
