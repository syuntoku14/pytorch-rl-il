import unittest
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from rlil import nn
from rlil.agents import LazyAgent
from rlil.environments import GymEnvironment, Action
from rlil.policies.deterministic import DeterministicPolicyNetwork
from rlil.samplers import ParallelEnvSampler


class MockLazyAgent(LazyAgent):
    def act(self, states, reward=None):
        self._states = states
        actions = self.models["policy"].eval(states.to(self.device))
        actions += self._noise.sample([actions.shape[0]])
        self._actions = Action(actions).to("cpu")
        return self._actions


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def test_sampler(self):
        env = gym.make('LunarLanderContinuous-v2')
        env = GymEnvironment(env)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.state_space.shape[0],
                      Action.action_space().shape[0])
        )
        model = DeterministicPolicyNetwork(model, Action.action_space())
        model.share_memory()
        models = {"policy": model}

        sampler = ParallelEnvSampler(
            MockLazyAgent,
            models, 
            env,
            n_workers=3,
            seed=0
        )
