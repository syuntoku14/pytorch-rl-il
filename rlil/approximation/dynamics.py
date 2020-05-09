import torch
from rlil.environments import State
from rlil.nn import RLNetwork
from .approximation import Approximation


class Dynamics(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='dynamics',
            **kwargs
    ):
        model = DynamicsModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class DynamicsModule(RLNetwork):
    def forward(self, states, actions):
        x = torch.cat((states.features.float(),
                       actions.features.float()), dim=1)
        diff_features = self.model(x)
        next_features = states.features + diff_features

        return State(
            next_features,
            mask=states.mask,
            info=states.info
        )
