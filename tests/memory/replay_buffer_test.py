import pytest
import random
import torch
import numpy as np
import gym
import torch_testing as tt
from rlil.environments import State, Action, GymEnvironment
from rlil.utils import Samples
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
        samples = Samples(state, action, rewards[i].unsqueeze(0), next_state)
        replay_buffer.store(samples)
        (s, a, r, n, w, i) = replay_buffer.sample(3)


def test_multi_store():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)
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
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)
    all_samples = replay_buffer.get_all_transitions()


def test_clear():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env)

    states = torch.tensor([env.observation_space.sample()]*20)
    actions = torch.tensor([env.action_space.sample()]*19)
    rewards = torch.arange(0, 19, dtype=torch.float)

    states = State(states)
    actions = Action(actions)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)
    assert len(replay_buffer) > 0
    replay_buffer.clear()
    assert len(replay_buffer) == 0


def test_n_step_run():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(
        5, env, n_step=3, discount_factor=0.9)

    states = torch.tensor([env.observation_space.sample()]*4)
    actions = torch.tensor([env.action_space.sample()]*3)
    rewards = torch.ones(3, dtype=torch.float)

    state = State(states[:-1])
    next_state = State(states[1:])
    action = Action(actions)
    samples = Samples(state, action, rewards, next_state)
    replay_buffer.store(samples)
    replay_buffer.on_episode_end()
    s, a, r, n, w, i = replay_buffer.get_all_transitions()
    tt.assert_equal(r.cpu(), torch.tensor([2.71, 1.9, 1.0], dtype=torch.float32))


def test_per_run():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    replay_buffer = ExperienceReplayBuffer(10000, env, prioritized=True)
    assert replay_buffer.prioritized

    # prioritized
    states = State(torch.tensor([env.observation_space.sample()]*4))
    actions = Action(torch.tensor([env.action_space.sample()]*3))
    rewards = torch.ones(3, dtype=torch.float)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    priorities = torch.ones(rewards.shape[0])
    replay_buffer.store(samples, priorities=priorities)

    # not prioritized
    states = State(torch.tensor([env.observation_space.sample()]*11))
    actions = Action(torch.tensor([env.action_space.sample()]*10))
    rewards = torch.zeros(10, dtype=torch.float)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    priorities = torch.zeros(rewards.shape[0])
    replay_buffer.store(samples, priorities=priorities)

    (s, a, r, n, w, i) = replay_buffer.sample(3)
    assert r.sum() == 3
    assert w.sum() < 3

    td_error = torch.zeros(3)
    replay_buffer.update_priorities(i, td_error.cpu())
    (s, a, r, n, w, i) = replay_buffer.sample(3)
    assert r.sum() < 3
    assert w.sum() == 3.
