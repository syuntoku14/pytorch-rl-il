import pytest
import torch
import gym
from torch import nn
from torch.nn.functional import mse_loss
import torch_testing as tt
import numpy as np
from rlil.environments import State, Action
from rlil.approximation import QNetwork, FixedTarget

STATE_DIM = 2
ACTIONS = 3
action_space = gym.spaces.Discrete(3)


@pytest.fixture
def setUp():
    torch.manual_seed(2)
    Action.set_action_space(action_space)
    model = nn.Sequential(
        nn.Linear(STATE_DIM, ACTIONS)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    q = QNetwork(model, optimizer)

    yield q


def test_eval_list(setUp):
    q = setUp
    states = State(
        torch.randn(5, STATE_DIM),
        mask=torch.tensor([1, 1, 0, 1, 0])
    )
    result = q.eval(states)
    assert result.shape == (5, 3)
    tt.assert_almost_equal(
        result,
        torch.tensor([
            [-0.238509, -0.726287, -0.034026],
            [-0.35688755, -0.6612102, 0.34849477],
            [0., 0., 0.],
            [0.1944, -0.5536, -0.2345],
            [0., 0., 0.]
        ]),
        decimal=2
    )


def test_eval_actions(setUp):
    q = setUp
    states = State(torch.randn(4, STATE_DIM))
    actions = Action(torch.tensor([1, 2, 0, 1]).unsqueeze(1))
    result = q.eval(states, actions)
    assert result.shape == (4, )


def test_reinforce(setUp):
    q = setUp
    states = State(torch.randn(4, STATE_DIM))
    actions = Action(torch.tensor([1, 2, 0, 1]).unsqueeze(1))
    targets = q.target(states, actions)
    targets += torch.randn(4)

    loss_old = mse_loss(q(states, actions), targets)
    for _ in range(10):
        loss = mse_loss(q(states, actions), targets)
        q.reinforce(loss)
    loss_new = mse_loss(q(states, actions), targets)

    assert loss_old > loss_new
