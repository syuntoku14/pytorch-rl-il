import unittest
import numpy as np
from rlil.environments.gym import GymEnvironment
from rlil.environments import State, Action
import torch
import gym


class GymEnvironmentTest(unittest.TestCase):
    def test_discrete(self):
        env = gym.make('CartPole-v0')
        env = GymEnvironment(env)
        env.reset()
        while not env._state.done:
            action = env.action_space.sample()
            action = Action(torch.tensor([action]).unsqueeze(0))
            state, reward = env.step(action)

    def test_continuous(self):
        env = gym.make('LunarLanderContinuous-v2')
        env = GymEnvironment(env)
        env.reset()
        while not env._state.done:
            action = env.action_space.sample()
            action = Action(torch.tensor([action]).unsqueeze(0))
            state, reward = env.step(action)