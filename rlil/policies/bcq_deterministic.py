import torch
from rlil.environments import squash_action
from rlil.approximation import Approximation
from rlil.nn import RLNetwork


class BCQDeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            phi=0.05,
            name='policy',
            **kwargs
    ):
        model = BCQDeterministicPolicyNetwork(model, space, phi)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class BCQDeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space, phi=0.05):
        super().__init__(model)
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2,
            dtype=torch.float32, device=self.device)
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2,
            dtype=torch.float32, device=self.device)
        self.phi = phi

    def forward(self, states, vae_actions):
        x = torch.cat((states.features.float(),
                       vae_actions.features.float()), dim=1)
        actions = self.model(x) * states.mask.float().unsqueeze(-1)
        actions = self.phi * squash_action(actions, self._tanh_scale, self._tanh_mean)
        return vae_actions.features + actions

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
