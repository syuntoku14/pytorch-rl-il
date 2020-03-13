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
    def forward(self, states, actions_raw=None):
        values = super().forward(states)
        if actions_raw is None:
            return values
        if isinstance(actions_raw, list):
            actions_raw = torch.tensor(actions_raw, device=self.device)
        return values.gather(1, actions_raw.view(-1, 1)).squeeze(1)
