import pytest
import numpy as np
import torch
from torch import nn
import torch_testing as tt
from gym.spaces import Box
from rlil.environments import State
from rlil.policies import SoftDeterministicPolicy

STATE_DIM = 2
ACTION_DIM = 3


@pytest.fixture
def setUp():
    torch.manual_seed(2)
    space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    model = nn.Sequential(
        nn.Linear(STATE_DIM, ACTION_DIM * 2)
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    policy = SoftDeterministicPolicy(model, optimizer, space)
    yield policy


def test_output_shape(setUp):
    policy = setUp
    state = State(torch.randn(1, STATE_DIM))
    action, _ = policy(state)
    assert action.shape == (1, ACTION_DIM)

    state = State(torch.randn(5, STATE_DIM))
    action, _ = policy(state)
    assert action.shape == (5, ACTION_DIM)


def test_reinforce_one(setUp):
    policy = setUp
    state = State(torch.randn(1, STATE_DIM))
    action, log_prob1 = policy(state)
    loss = -log_prob1.mean()
    policy.reinforce(loss)

    action, log_prob2 = policy(state)

    assert log_prob2.item() > log_prob1.item()


def test_sample_multiple(setUp):
    policy = setUp
    state = State(torch.randn(5, STATE_DIM))
    actions, raw_actions = policy.sample_multiple(state, num_sample=10)
    assert actions.shape == (5, 10, ACTION_DIM)
    assert raw_actions.shape == (5, 10, ACTION_DIM)
