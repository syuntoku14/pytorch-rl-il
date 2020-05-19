import pytest
import numpy as np
import torch
import torch_testing as tt
import gym
from rlil import nn
from rlil.environments import State


@pytest.fixture
def setUp():
    torch.manual_seed(2)


def test_dueling(setUp):
    torch.random.manual_seed(0)
    value_model = nn.Linear(2, 1)
    advantage_model = nn.Linear(2, 3)
    model = nn.Dueling(value_model, advantage_model)
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = model(states).detach().numpy()
    np.testing.assert_array_almost_equal(
        result,
        np.array(
            [[-0.495295, 0.330573, 0.678836], [-1.253222, 1.509323, 2.502186]],
            dtype=np.float32,
        ),
    )


def test_linear0(setUp):
    model = nn.Linear0(3, 3)
    result = model(torch.tensor([[3.0, -2.0, 10]]))
    tt.assert_equal(result, torch.tensor([[0.0, 0.0, 0.0]]))


def test_list(setUp):
    model = nn.Linear(2, 2)
    net = nn.RLNetwork(model, (2,))
    features = torch.randn((4, 2))
    done = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    out = net(State(features, done))
    tt.assert_almost_equal(
        out,
        torch.tensor(
            [
                [0.0479387, -0.2268031],
                [0.2346841, 0.0743403],
                [0.0, 0.0],
                [0.2204496, 0.086818],
            ]
        ),
    )

    features = torch.randn(3, 2)
    done = torch.tensor([1, 1, 1], dtype=torch.bool)
    out = net(State(features, done))
    tt.assert_almost_equal(
        out,
        torch.tensor(
            [
                [0.4234636, 0.1039939],
                [0.6514298, 0.3354351],
                [-0.2543002, -0.2041451],
            ]
        ),
    )


def test_categorical_dueling(setUp):
    n_actions = 2
    n_atoms = 3
    value_model = nn.Linear(2, n_atoms)
    advantage_model = nn.Linear(2, n_actions * n_atoms)
    model = nn.CategoricalDueling(value_model, advantage_model)
    x = torch.randn((2, 2))
    out = model(x)
    assert out.shape == (2, 6)
    tt.assert_almost_equal(
        out,
        torch.tensor(
            [
                [0.014, -0.691, 0.251, -0.055, -0.419, -0.03],
                [0.057, -1.172, 0.568, -0.868, -0.482, -0.679],
            ]
        ),
        decimal=3,
    )


def help_perturb(model, inputs):
    # noisy net perturbs the parameter every episodes, not every actions
    result1 = model(inputs)
    result2 = model(inputs)
    tt.assert_equal(result1, result2)

    # perturbed model
    model.perturb()
    result3 = model(inputs)
    with pytest.raises(AssertionError):
        tt.assert_equal(result1, result3)


def test_noisy_linear(setUp):
    model = nn.NoisyLinear(3, 3)
    inputs = torch.tensor([[3.0, -2.0, 10]])
    help_perturb(model, inputs)


def test_noisy_factorized_linear(setUp):
    model = nn.NoisyFactorizedLinear(3, 3)
    inputs = torch.tensor([[3.0, -2.0, 10]])
    help_perturb(model, inputs)


def help_perturb_noisy_layers(model, inputs):
    # noisy net perturbs the parameter every episodes, not every actions
    result1 = model(inputs)
    result2 = model(inputs)
    tt.assert_equal(result1, result2)

    # perturbed model
    model.apply(nn.perturb_noisy_layers)
    result3 = model(inputs)
    with pytest.raises(AssertionError):
        tt.assert_equal(result1, result3)


def test_perturb_noisy_layers(setUp):
    inputs = torch.tensor([[3.0, -2.0, 10]])
    model1 = nn.Sequential(
        nn.Linear(3, 3),
        nn.NoisyFactorizedLinear(3, 3),
    )
    help_perturb_noisy_layers(model1, inputs)

    model2 = nn.Sequential(
        nn.Linear(3, 3),
        nn.NoisyLinear(3, 3),
    )
    help_perturb_noisy_layers(model2, inputs)

    model3 = nn.RLNetwork(nn.Sequential(
        nn.Linear(3, 3),
        nn.NoisyLinear(3, 3),
    ))
    help_perturb_noisy_layers(model3, State(inputs))


def test_mmd(setUp):
    batch_size = 10
    sample_size = 5
    dimension = 3

    sample_actions1 = torch.randn([batch_size, sample_size, dimension])
    sample_actions2 = torch.randn([batch_size, sample_size, dimension])
    nn.mmd_loss_laplacian(sample_actions1, sample_actions2)
    nn.mmd_loss_gaussian(sample_actions1, sample_actions2)


def assert_array_equal(actual, expected):
    for first, second in zip(actual, expected):
        if second is None:
            assertIsNone(first)
        else:
            tt.assert_almost_equal(first, second, decimal=3)
