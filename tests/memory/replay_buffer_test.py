import pytest
import random
import torch
import numpy as np
import gym
import torch_testing as tt
from rlil.environments import State, Action, GymEnvironment
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import set_device


def test_run():
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    for i in range(10):
        state = State(states[i].view(1, -1), torch.tensor([1]).bool())
        next_state = State(
            states[i + 1].view(1, -1), torch.tensor([1]).bool())
        action = Action(actions[i].view(1, -1))
        replay_buffer.store(
            state, action, rewards[i].unsqueeze(0), next_state)
        sample = replay_buffer.sample(3)


def test_multi_store():
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])
    for i in range(2):
        sample = replay_buffer.sample(3)


def test_clear():
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])
    replay_buffer.clear()
    assert len(replay_buffer) == 0
