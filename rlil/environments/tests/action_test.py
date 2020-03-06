import unittest
import numpy as np
import torch
import torch_testing as tt
from rlil.environments.action import Action, action_decorator


class ActionTest(unittest.TestCase):
    def test_constructor_defaults(self):
        raw = torch.randn(3, 4)
        action = Action(raw)
        tt.assert_equal(action.raw, raw)

    def test_from_list(self):
        action1 = Action(torch.randn(1, 4))
        action2 = Action(torch.randn(1, 4))
        action3 = Action(torch.randn(1, 4))
        action = Action.from_list([action1, action2, action3])
        tt.assert_equal(action.raw, torch.cat(
            (action1.raw, action2.raw, action3.raw)))

    def test_from_numpy(self):
        actions = np.array([[1, 2, 3]])
        action = Action.from_numpy(actions)
        tt.assert_equal(action.raw, torch.tensor([[1, 2, 3]]))

    def test_get_item(self):
        raw = torch.randn(3, 4)
        actions = Action(raw)
        action = actions[2]
        tt.assert_equal(action.raw, raw[2].unsqueeze(0))

    def test_len(self):
        action = Action(torch.randn(3, 4))
        self.assertEqual(len(action), 3)

    def test_action_decorator(self):
        @action_decorator
        def act():
            return torch.tensor([3, 4]).unsqueeze(0)
        
        action = act()
        tt.assert_equal(action.raw, torch.tensor([3, 4]).unsqueeze(0))
