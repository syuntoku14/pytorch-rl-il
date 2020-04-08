import pytest
import numpy as np
from rlil.environments.gym import GymEnvironment
from rlil.environments import State, Action
import torch
import gym


def test_env_discrete():
    env = gym.make('CartPole-v0')
    env = GymEnvironment(env)
    env.reset()
    while not env._state.done:
        action = Action.action_space().sample()
        action = Action(torch.tensor([action]).unsqueeze(0))
        state, reward = env.step(action)


def test_env_continuous():
    env = gym.make('LunarLanderContinuous-v2')
    env = GymEnvironment(env)
    env.reset()
    while not env._state.done:
        action = Action.action_space().sample()
        action = Action(torch.tensor([action]))
        state, reward = env.step(action)
