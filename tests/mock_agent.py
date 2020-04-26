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
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import get_replay_buffer


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
        self.replay_buffer.store(self._state,
                                 self._action,
                                 reward,
                                 state)
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
        self._replay_buffer = None

    def set_replay_buffer(self, env):
        self._replay_buffer = ExperienceReplayBuffer(1e9, env)

    def act(self, state, reward):
        self._replay_buffer.store(self._state,
                                  self._action,
                                  reward,
                                  state)
        self._state = state

        with torch.no_grad():
            action = self.policy_model(
                state.to(self.policy_model.device))

        self._action = Action(action).to("cpu")
        return self._action
