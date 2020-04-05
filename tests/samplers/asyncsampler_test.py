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


class MockLazyAgent:
    def __init__(self, env):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.state_space.shape[0],
                      Action.action_space().shape[0])
        )
        self.policy_model = DeterministicPolicyNetwork(
            model, Action.action_space())

        self._state = None
        self._action = None
        self.buffer = {"states": [],
                       "actions": [],
                       "rewards": [],
                       "next_states": []}

    def act(self, state, reward):
        if self._state is not None and self._action is not None:
            self.buffer["states"].append(self._state)
            self.buffer["actions"].append(self._action)
            self.buffer["rewards"].append(reward)
            self.buffer["next_states"].append(state)

        self._state = state

        with torch.no_grad():
            action = self.policy_model(
                state.to(self.policy_model.device))

        self._action = Action(action).to("cpu")
        return self._action


class TestSampler(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        ray.init(include_webui=False, ignore_reinit_error=True)

        self.replay_buffer_size = 100
        replay_buffer = ExperienceReplayBuffer(self.replay_buffer_size)
        set_replay_buffer(replay_buffer)
        self.env = GymEnvironment('LunarLanderContinuous-v2')
        self.lazy_agent = MockLazyAgent(self.env)
        self.sampler = AsyncSampler(
            self.env,
            num_workers=3,
            seed=0,
        )
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f sec' % (self.id(), t))

    def test_sampler_episode(self):
        max_episodes = 6
        self.sampler.start_sampling(self.lazy_agent, max_episodes=max_episodes)
        self.sampler.store_samples(timeout=1e8)

        for worker in self.sampler._workers:
            assert ray.get(worker.episodes.remote()) >= max_episodes
        assert len(self.sampler._replay_buffer) == self.replay_buffer_size

    def test_sampler_frames(self):
        max_frames = 50

        self.sampler.start_sampling(self.lazy_agent, max_frames=max_frames)
        self.sampler.store_samples(timeout=1e8)

        for worker in self.sampler._workers:
            assert ray.get(worker.frames.remote()) >= max_frames
        assert len(self.sampler._replay_buffer) == self.replay_buffer_size

    def test_ray_wait(self):
        max_episodes = 100

        self.sampler.start_sampling(self.lazy_agent, max_episodes=max_episodes)
        self.sampler.store_samples()

        assert len(self.sampler._replay_buffer) == 0


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(SomeTest)
    unittest.TextTestRunner(verbosity=0).run(suite)
