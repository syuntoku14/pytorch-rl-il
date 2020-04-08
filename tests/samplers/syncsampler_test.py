import pytest
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


@pytest.mark.skip()
def test_sampler_synchronous():
    ray.init(include_webui=False, ignore_reinit_error=True)

    replay_buffer = ExperienceReplayBuffer(1e6)
    set_replay_buffer(replay_buffer)
    env = GymEnvironment('LunarLanderContinuous-v2')
    agent = MockAgent(self.env)
    sampler = SyncSampler(env, num_workers=3, seed=0)

    lazy_agent = agent.make_lazy_agent()
    sum_frames = 0
    sum_episodes = 0
    while sum_episodes == 0:
        self.sampler.start_sampling(lazy_agent)
        frames, episodes = self.sampler.store_samples()
        sum_frames += frames
        sum_episodes += episodes

    assert len(self.sampler._replay_buffer) == sum_frames
