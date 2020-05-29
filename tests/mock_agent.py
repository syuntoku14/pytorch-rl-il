import unittest
import numpy as np
import torch
from rlil import nn
from rlil.environments import Action
from rlil.policies.deterministic import DeterministicPolicyNetwork
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import get_replay_buffer, get_n_step
from rlil.utils import Samples


class MockAgent:
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
        self.replay_buffer = get_replay_buffer()

    def act(self, state, reward):
        samples = Samples(self._state, self._action, reward, state)
        self.replay_buffer.store(samples)
        self._state = state

        with torch.no_grad():
            action = self.policy_model(
                state.to(self.policy_model.device))

        self._action = Action(action).to("cpu")
        return self._action

    def make_lazy_agent(self):
        return MockLazyAgent(self.policy_model)

    def train(self):
        pass


class MockLazyAgent:
    def __init__(self, policy_model):
        self._state = None
        self._action = None
        self.policy_model = policy_model
        self.replay_buffer = None
        # for N step replay buffer
        self._n_step, self._discount_factor = get_n_step()

    def set_replay_buffer(self, env):
        self.replay_buffer = ExperienceReplayBuffer(
            1e7, env, n_step=self._n_step,
            discount_factor=self._discount_factor)

    def act(self, state, reward):
        samples = Samples(self._state, self._action, reward, state)
        self.replay_buffer.store(samples)
        self._state = state

        with torch.no_grad():
            action = self.policy_model(
                state.to(self.policy_model.device))

        self._action = Action(action).to("cpu")
        return self._action
