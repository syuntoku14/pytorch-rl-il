import torch
from rlil.nn import RLNetwork
from .approximation import Approximation


class EnsembleQContinuous(Approximation):
    def __init__(
            self,
            models: torch.nn.ModuleList,
            optimizer,
            name='ensemble_q',
            **kwargs
    ):
        model = EnsembleQContinuousModule(models)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def q1(self, *args, **kwargs):
        return self.model.q1(*args, **kwargs)


class EnsembleQContinuousModule(RLNetwork):
    def forward(self, states, actions):
        all_qs = []
        x = torch.cat((states.features.float(),
                       actions.features.float()), dim=1)
        for m in self.model:
            all_qs.append((m(x).squeeze(-1)
                           * states.mask.float()).unsqueeze(1))
        all_qs = torch.cat(all_qs, dim=1)
        return all_qs  # batch x num_q

    def q1(self, states, actions):
        x = torch.cat((states.features.float(),
                       actions.features.float()), dim=1)
        return self.model[0](x).squeeze(-1) * states.mask.float()
