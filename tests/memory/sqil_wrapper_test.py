import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from rlil.environments import State, Action, GymEnvironment
from rlil.memory import ExperienceReplayBuffer, SqilWrapper
from rlil.initializer import set_device


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(1000, env)

    # base buffer
    states = State(torch.tensor([env.observation_space.sample()]*10))
    actions = Action(torch.tensor([env.action_space.sample()]*9))
    rewards = torch.arange(0, 9, dtype=torch.float)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])

    # expert buffer
    exp_replay_buffer = ExperienceReplayBuffer(1000, env)
    exp_states = State(torch.tensor([env.observation_space.sample()]*10))
    exp_actions = Action(torch.tensor([env.action_space.sample()]*9))
    exp_rewards = torch.arange(10, 19, dtype=torch.float)
    exp_replay_buffer.store(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])

    sqil_buffer = SqilWrapper(replay_buffer, exp_replay_buffer)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    yield sqil_buffer, samples


def test_sample(setUp):
    sqil_buffer, samples = setUp
    res_states, res_actions, res_rewards, res_next_states, _ = \
        sqil_buffer.sample(40)

    # test rewards
    # half of the rewards are 1 and the others are 0
    assert res_rewards.sum() == len(res_rewards) / 2
