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

    def sample_multiple(self, states, num_sample=10):
        return self.model.sample_multiple(states, num_sample)


class SoftDeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2,
            dtype=torch.float32, device=self.device)
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2,
            dtype=torch.float32, device=self.device)

    def forward(self, state, return_mean=False):
        outputs = super().forward(state)
        if return_mean:
            means = outputs[:, 0: self._action_dim]
            means = squash_action(means, self._tanh_scale, self._tanh_mean)
            return means

        # make normal distribution
        means = outputs[:, 0: self._action_dim]
        logvars = outputs[:, self._action_dim:]
        std = logvars.mul(0.5).exp_()
        normal = torch.distributions.normal.Normal(means, std)

        # sample from the normal distribution
        # see openai spinningup for log_prob computation:
        # https://github.com/openai/spinningup/blob/e76f3cc1dfbf94fe052a36082dbd724682f0e8fd/spinup/algos/pytorch/sac/core.py#L53
        raw = normal.rsample()
        log_prob = normal.log_prob(raw).sum(axis=-1)
        log_prob -= (2*(np.log(2) - raw - F.softplus(-2*raw))).sum(axis=1)

        action = squash_action(raw, self._tanh_scale, self._tanh_mean)
        return action, log_prob

    def sample_multiple(self, state, num_sample=10):
        # this function is used in BEAR training
        outputs = super().forward(state)

        # make normal distribution
        means = outputs[:, 0: self._action_dim]
        repeated_means = torch.repeat_interleave(
            means.unsqueeze(1), num_sample, 1)
        logvars = outputs[:, self._action_dim:]
        repeated_logvars = torch.repeat_interleave(
            logvars.unsqueeze(1), num_sample, 1)
        repeated_std = repeated_logvars.mul(0.5).exp_()
        # batch x num_sample x d
        normal = torch.distributions.normal.Normal(
            repeated_means, repeated_std)
        raw = normal.rsample()

        action = squash_action(raw, self._tanh_scale, self._tanh_mean)
        return action, raw

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
