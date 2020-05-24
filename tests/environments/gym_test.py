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


def test_append_time():
    env = gym.make('LunarLanderContinuous-v2')
    env = GymEnvironment(env, append_time=True)
    state = env.reset()
    last_timestep = state.raw[0, -1].item()
    while not env._state.done:
        action = Action.action_space().sample()
        action = Action(torch.tensor([action]))
        state, reward = env.step(action)
        assert state.raw[0, -1].item() > last_timestep
        last_timestep = state.raw[0, -1].item()
    assert state.shape[1] == env._env.observation_space.shape[0] + 1
