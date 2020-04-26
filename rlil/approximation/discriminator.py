import torch
from rlil.nn import RLNetwork
from .approximation import Approximation


class Discriminator(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='discriminator',
            **kwargs
    ):
        model = DiscriminatorModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def expert_reward(self, features):
        return -torch.log(self.model(features)).squeeze().detach()


class DiscriminatorModule(RLNetwork):
    def forward(self, features):
        return self.model(features)
