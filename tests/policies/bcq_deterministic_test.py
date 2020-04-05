import unittest
import torch
import torch_testing as tt
import numpy as np
from gym.spaces import Box
from rlil import nn
from rlil.approximation import FixedTarget
from rlil.environments import State, Action, squash_action
from rlil.policies import BCQDeterministicPolicy

# import ptvsd
# ptvsd.enable_attach()
# ptvsd.wait_for_attach()

STATE_DIM = 2
ACTION_DIM = 3


class TestBCQDeterministic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear0(STATE_DIM + ACTION_DIM, ACTION_DIM)
        )
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.space = Box(np.array([-1, -1, -1]),
                         np.array([1, 1, 1]), dtype=np.float32)
        self.policy = BCQDeterministicPolicy(
            self.model,
            self.optimizer,
            self.space
        )
        Action.set_action_space(self.space)

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        vae_action = Action(torch.randn(1, ACTION_DIM))
        action = self.policy(state, vae_action)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        vae_action = Action(torch.randn(5, ACTION_DIM))
        action = self.policy(state, vae_action)
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_step_one(self):
        state = State(torch.randn(1, STATE_DIM))
        vae_action = Action(torch.randn(1, ACTION_DIM))
        self.policy(state, vae_action)
        self.policy.step()

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        vae_action = Action(torch.randn(1, ACTION_DIM))
        target = vae_action.features + torch.tensor([[0.25, 0.5, -0.5]])

        for _ in range(0, 200):
            action = self.policy(state, vae_action)
            loss = ((target - action) ** 2).mean()
            loss.backward()
            self.policy.step()

        self.assertLess(loss, 0.001)

    def test_target(self):
        self.policy = BCQDeterministicPolicy(
            self.model,
            self.optimizer,
            self.space,
            target=FixedTarget(3)
        )

        # choose initial action
        state = State(torch.ones(1, STATE_DIM))
        vae_action = Action(torch.ones(1, ACTION_DIM))
        action = self.policy(state, vae_action)
        tt.assert_equal(action, squash_action(
            vae_action.features, action_space=self.space))

        # run update step, make sure target network doesn't change
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.target(state, vae_action),
                        squash_action(vae_action.features, action_space=self.space))

        # again...
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.target(state, vae_action),
                        squash_action(vae_action.features, action_space=self.space))

        # third time, target should be updated
        action.sum().backward(retain_graph=True)
        self.policy.step()
        # tt.assert_allclose(
        #     self.policy.eval(state, vae_action),
        #     torch.tensor([[-0.595883, -0.595883, -0.595883]]),
        #     atol=1e-4,
        # )


if __name__ == '__main__':
    unittest.main()
