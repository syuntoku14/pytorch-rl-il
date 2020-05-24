import pytest
import torch
import torch_testing as tt
from torch.nn.functional import mse_loss
from rlil import nn
from rlil.approximation.ensemble_q_continuous import EnsembleQContinuous
from rlil.environments import State, Action, GymEnvironment
from rlil.presets.continuous.models import fc_q
import numpy as np


@pytest.fixture
def setUp():
    env = GymEnvironment('LunarLanderContinuous-v2', append_time=True)
    num_qs = 3
    num_samples = 5
    q_models = nn.ModuleList([fc_q(env) for _ in range(num_qs)])
    qs_optimizer = torch.optim.Adam(q_models.parameters())
    qs = EnsembleQContinuous(q_models, qs_optimizer)
    Action.set_action_space(env.action_space)
    sample_states = State.from_list([env.reset() for _ in range(num_samples)])
    sample_actions = Action(
        torch.tensor([env.action_space.sample() for _ in range(num_samples)]))

    yield qs, sample_states, sample_actions


def test_forward(setUp):
    qs, states, actions = setUp
    q_values = qs(states, actions)
    assert q_values.shape == (5, 3)
    with pytest.raises(AssertionError):
        tt.assert_almost_equal(q_values[0][0], q_values[0][1])
        tt.assert_almost_equal(q_values[0][0], q_values[0][2])


def test_q1(setUp):
    qs, states, actions = setUp
    q_values = qs.q1(states, actions)
    assert q_values.shape == (5, )


def test_reinforce(setUp):
    qs, states, actions = setUp
    q_values = qs(states, actions)
    qs_params = [param.data.clone() for param in qs.model.parameters()]
    qs.reinforce(q_values.sum())
    new_qs_params = [param.data for param in qs.model.parameters()]

    for param, new_param in zip(qs_params, new_qs_params):
        with pytest.raises(AssertionError):
            tt.assert_almost_equal(param, new_param)
