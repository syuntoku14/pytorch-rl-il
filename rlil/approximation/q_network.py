import torch
from rlil.nn import RLNetwork
from .approximation import Approximation


class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='q',
            **kwargs
    ):
        model = QModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class QModule(RLNetwork):
    def forward(self, states, actions=None):
        values = super().forward(states)
        if actions is None:
            return values
        return values.gather(1, actions.features.view(-1, 1)).squeeze(1)
