import torch
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from rlil.environments import squash_action
from rlil.approximation import Approximation
from rlil.nn import RLNetwork


class GaussianPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name='policy',
            **kwargs
    ):
        super().__init__(
            GaussianPolicyNetwork(model, space),
            optimizer,
            name=name,
            **kwargs
        )


class GaussianPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]

    def forward(self, state, return_mean=False):
        outputs = super().forward(state)
        means = outputs[:, :self._action_dim]

        if return_mean:
            return means

        logvars = outputs[:, self._action_dim:]
        std = logvars.exp_()
        return Independent(Normal(means, std), 1)

    def to(self, device):
        return super().to(device)
