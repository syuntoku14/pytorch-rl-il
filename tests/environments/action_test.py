import unittest
import numpy as np
import torch
import torch_testing as tt
import gym
from rlil.environments.action import Action, action_decorator

# TODO: better pytorch testing


class ActionTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

    def test_set_action_space(self):
        with self.assertRaises(AssertionError, msg="action_space is not set. Use Action.set_action_space function."):
            Action(torch.Tensor([[2, 3]]))

    def test_continuous_action(self):
        action_space = gym.spaces.Box(
            low=np.array([-1, -10]), high=np.array([1, 10]))
        Action.set_action_space(action_space)
        raw = torch.FloatTensor([[0, 0], [2, 2], [-20, -20]])
        action = Action(raw)
        tt.assert_equal(action.raw, raw)
        # test action clip
        tt.assert_equal(action.features, torch.FloatTensor(
            [[0, 0], [1, 2], [-1, -10]]))
        with self.assertRaises(AssertionError, msg="Action.raw.shape torch.Size([3, 5]) is invalid. It doesn't match the action_space."):
            raw = torch.randn(3, 5)
            action = Action(raw)

    def test_discrete_action(self):
        action_space = gym.spaces.Discrete(4)
        Action.set_action_space(action_space)
        raw = torch.tensor([1, 2, 3, 0]).unsqueeze(1)
        action = Action(raw)
        tt.assert_equal(action.raw, raw)
        with self.assertRaises(AssertionError, msg="Invalid action value"):
            raw = torch.tensor([5])
            action = Action(raw)

    def test_from_list(self):
        action_space = gym.spaces.Box(low=np.array(
            [-1, -2, -3, -4]), high=np.array([1, 2, 3, 4]))
        Action.set_action_space(action_space)
        action1 = Action(torch.randn(1, 4))
        action2 = Action(torch.randn(1, 4))
        action3 = Action(torch.randn(1, 4))
        action = Action.from_list([action1, action2, action3])
        tt.assert_equal(action.raw, torch.cat(
            (action1.raw, action2.raw, action3.raw)))

    def test_from_numpy(self):
        action_space = gym.spaces.Box(low=0, high=5, shape=(3, ))
        Action.set_action_space(action_space)
        actions = np.array([[1, 2, 3]])
        action = Action.from_numpy(actions)
        tt.assert_equal(action.raw, torch.tensor([[1, 2, 3]]))

    def test_get_item(self):
        action_space = gym.spaces.Box(low=np.array(
            [-1, -2, -3, -4]), high=np.array([1, 2, 3, 4]))
        Action.set_action_space(action_space)
        raw = torch.randn(3, 4)
        actions = Action(raw)
        action = actions[2]
        tt.assert_equal(action.raw, raw[2].unsqueeze(0))

    def test_len(self):
        action_space = gym.spaces.Box(low=np.array(
            [-1, -2, -3, -4]), high=np.array([1, 2, 3, 4]))
        Action.set_action_space(action_space)
        action = Action(torch.randn(3, 4))
        self.assertEqual(len(action), 3)

    def test_action_decorator(self):
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))
        Action.set_action_space(action_space)
        @action_decorator
        def act():
            return torch.tensor([3, 4]).unsqueeze(0)

        action = act()
        tt.assert_equal(action.raw, torch.tensor([3, 4]).unsqueeze(0))
