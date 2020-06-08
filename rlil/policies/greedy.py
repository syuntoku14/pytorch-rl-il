import numpy as np
import torch
from rlil.approximation.q_network import QModule


class GreedyPolicy:
    def __init__(
            self,
            q_model,
            num_actions,
            epsilon: float = 0.,
    ):
        if isinstance(q_model, QModule):
            self.model = q_model
        else:
            self.model = QModule(q_model)
        self.num_actions = num_actions
        self.epsilon = epsilon

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, states):
        random_flags = torch.rand(
            len(states), device=self.model.device) < self.epsilon
        random_actions = torch.randint(
            self.num_actions, (len(states),), device=self.model.device)
        greedy_actions = torch.argmax(self.model(states), dim=1)
        actions = torch.where(random_flags, random_actions, greedy_actions)
        return actions.unsqueeze(1)

    def no_grad(self, states):
        with torch.no_grad():
            return self.__call__(states)

    def eval(self, states):
        with torch.no_grad():
            # check current mode
            mode = self.model.training
            # switch to eval mode
            self.model.eval()
            # run forward pass
            result = self.model(states)
            # change to original mode
            self.model.train(mode)
        return torch.argmax(result, dim=1).unsqueeze(1)

    def to(self, device):
        self.model.to(device)
