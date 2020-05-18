import pytest
import numpy as np
import torch
import torch_testing as tt
import gym
from rlil.environments.action import Action, action_decorator


@pytest.fixture()
def set_continuous_action_space():
    action_space = gym.spaces.Box(
        low=np.array([-1, -10]), high=np.array([1, 10]))
    Action.set_action_space(action_space)


@pytest.fixture()
def set_discrete_action_space():
    action_space = gym.spaces.Discrete(4)
    Action.set_action_space(action_space)


def test_set_action_space_raises():
    """ 
    Action class should raise when the action_space is not set
    """
    with pytest.raises(AssertionError):
        Action(torch.Tensor([[2, 3]]))


def test_continuous_action(set_continuous_action_space):
    # GIVEN a set action_space

    # WHEN a new Action object with valid input is made
    # THEN the raw is equal to Action.raw
    raw = torch.tensor([[0, 0], [2, 2], [-20, -20]], dtype=torch.float32)
    action = Action(raw)
    tt.assert_equal(action.raw, raw)

    # WHEN a new Action object with a raw outside the action_space
    # THEN the action.features should clipped in the range
    tt.assert_equal(action.features, torch.tensor(
        [[0, 0], [1, 2], [-1, -10]], dtype=torch.float32))

    # WHEN a new Action object with invalid input is made
    # THEN raise a assertion error
    with pytest.raises(AssertionError):
        raw = torch.randn(3, 5)
        action = Action(raw)


def test_discrete_action(set_discrete_action_space):
    # GIVEN a set action_space

    # WHEN a new Action object with valid input is made
    # THEN the raw is equal to Action.raw
    raw = torch.tensor([1, 2, 3, 0]).unsqueeze(1)
    action = Action(raw)
    tt.assert_equal(action.raw, raw)

    # WHEN a new Action object with invalid input is made
    # THEN raise a assertion error
    with pytest.raises(AssertionError):
        raw = torch.tensor([5])
        action = Action(raw)


def test_from_list(set_continuous_action_space):
    action1 = Action(torch.randn(1, 2))
    action2 = Action(torch.randn(1, 2))
    action3 = Action(torch.randn(1, 2))
    action = Action.from_list([action1, action2, action3])
    tt.assert_equal(action.raw, torch.cat(
        (action1.raw, action2.raw, action3.raw)))


def test_from_numpy(set_continuous_action_space):
    actions = np.array([[1, 2]])
    action = Action.from_numpy(actions)
    tt.assert_equal(action.raw, torch.tensor([[1, 2]]))


def test_raw_numpy(set_continuous_action_space):
    actions = np.array([[1, 2]])
    action = Action.from_numpy(actions)
    np.testing.assert_equal(actions, action.raw_numpy())


def test_get_item():
    action_space = gym.spaces.Box(low=np.array(
        [-1, -2, -3, -4]), high=np.array([1, 2, 3, 4]))
    Action.set_action_space(action_space)
    raw = torch.randn(3, 4)
    actions = Action(raw)
    action = actions[2]
    tt.assert_equal(action.raw, raw[2].unsqueeze(0))


def test_len():
    action_space = gym.spaces.Box(low=np.array(
        [-1, -2, -3, -4]), high=np.array([1, 2, 3, 4]))
    Action.set_action_space(action_space)
    action = Action(torch.randn(3, 4))
    assert len(action) == 3


def test_action_decorator():
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))
    Action.set_action_space(action_space)
    @action_decorator
    def act():
        return torch.tensor([3, 4]).unsqueeze(0)

    action = act()
    tt.assert_equal(action.raw, torch.tensor([3, 4]).unsqueeze(0))
