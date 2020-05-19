import pytest
import torch
import torch_testing as tt
from torch.nn.functional import mse_loss
from rlil import nn
from rlil.approximation import Dynamics
from rlil.environments import State, Action, GymEnvironment
from rlil.presets.continuous.models import fc_dynamics


@pytest.fixture
def setUp():
    env = GymEnvironment('LunarLanderContinuous-v2')
    dynamics_model = fc_dynamics(env)

    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters())
    dynamics = Dynamics(model=dynamics_model,
                        optimizer=dynamics_optimizer)

    yield env, dynamics


def test_forward(setUp):
    env, dynamics = setUp
    state = env.reset()
    for _ in range(10):
        action = Action(
            torch.tensor(env.action_space.sample()).unsqueeze(0))
        output = dynamics(state, action)
        assert state.features.shape == output.features.shape
        tt.assert_equal(state.mask, output.mask)


def test_reinforce(setUp):
    env, dynamics = setUp
    state = env.reset()
    action = Action(
        torch.tensor(env.action_space.sample()).unsqueeze(0))
    output = dynamics(state, action)
    loss = mse_loss(state.features, output.features)
    for _ in range(10):
        output = dynamics(state, action)
        new_loss = mse_loss(state.features, output.features)
        dynamics.reinforce(new_loss)
    assert new_loss < loss
