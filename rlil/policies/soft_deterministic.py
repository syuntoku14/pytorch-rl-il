import torch
import numpy as np
import torch.nn.functional as F
from rlil.approximation import Approximation
from rlil.nn import RLNetwork
from rlil.environments import squash_action


class SoftDeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name="policy",
            **kwargs
    ):
        model = SoftDeterministicPolicyNetwork(model, space)
        super().__init__(model, optimizer, name=name, **kwargs)


class SoftDeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2).to(self.device)

    def forward(self, state, return_mean=False):
        outputs = super().forward(state)
        if return_mean:
            means = outputs[:, 0: self._action_dim]
            means = squash_action(means, self._tanh_scale, self._tanh_mean)
            return means

        normal = self._normal(outputs)
        action, log_prob = self._sample(normal)
        return action, log_prob

    def _normal(self, outputs):
        means = outputs[:, 0: self._action_dim]
        logvars = outputs[:, self._action_dim:]
        std = logvars.mul(0.5).exp_()
        return torch.distributions.normal.Normal(means, std)

    def _sample(self, normal):
        raw = normal.rsample()
        # see openai spinningup:
        # https://github.com/openai/spinningup/blob/e76f3cc1dfbf94fe052a36082dbd724682f0e8fd/spinup/algos/pytorch/sac/core.py#L53
        log_prob = normal.log_prob(raw).sum(axis=-1)
        log_prob -= (2*(np.log(2) - raw - F.softplus(-2*raw))).sum(axis=1)

        action = squash_action(raw, self._tanh_scale, self._tanh_mean)
        return action, log_prob

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
