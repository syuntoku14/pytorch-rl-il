import pytest
import numpy as np
import torch
from torch import nn
import torch_testing as tt
from gym.spaces import Discrete
from rlil.environments import State, Action
from rlil.policies import GreedyPolicy

STATE_DIM = 2
ACTIONS = 3
action_space = Discrete(ACTIONS)


@pytest.fixture
def setUp():
    torch.manual_seed(2)
    Action.set_action_space(action_space)
    model = nn.Sequential(
        nn.Linear(STATE_DIM, ACTIONS)
    )
    states = State(
        torch.randn(5, STATE_DIM),
        mask=torch.tensor([1, 1, 0, 1, 0])
    )
    yield model, states


def test_output_shape(setUp):
    model, states = setUp
    policy = GreedyPolicy(model, ACTIONS, 0.0)
    assert policy(states).shape == (states.shape[0], 1)
    assert policy.no_grad(states).shape == (states.shape[0], 1)
    policy.set_epsilon(1.0)
    assert policy.no_grad(states).shape == (states.shape[0], 1)
    assert policy.eval(states).shape == (states.shape[0], 1)


def test_epsilon(setUp):
    model, states = setUp
    policy = GreedyPolicy(model, ACTIONS, 1.0)
    random_actions = policy(states)
    with pytest.raises(AssertionError):
        tt.assert_almost_equal(
            random_actions,
            policy(states)
        )
    policy.set_epsilon(0.0)
    greedy_actions = policy(states)
    tt.assert_almost_equal(
        greedy_actions,
        policy(states)
    )
