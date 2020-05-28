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
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
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
        (s, a, r, n, w, i) = replay_buffer.sample(3)


def test_multi_store():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])
    for i in range(2):
        (s, a, r, n, w, i) = replay_buffer.sample(3)


def test_get_all_transitions():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])
    s, a, r, n = replay_buffer.get_all_transitions()


def test_clear():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])
    replay_buffer.clear()
    assert len(replay_buffer) == 0


def test_n_step_run():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(
        10000, env, n_step=3, discount_factor=0.9)

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
        replay_buffer.on_episode_end()
        (s, a, r, n, w, i) = replay_buffer.sample(3)


def test_per_run():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env, prioritized=True)
    assert replay_buffer.prioritized

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    priorities = torch.ones(rewards.shape[0])
    replay_buffer.store(states[:-1], actions, rewards,
                        states[1:], priorities=priorities)
    for i in range(2):
        (s, a, r, n, w, i) = replay_buffer.sample(3)
        td_error = s.features.sum(dim=1)
        assert td_error.shape == (3, )
        replay_buffer.update_priorities(i, td_error.cpu())
