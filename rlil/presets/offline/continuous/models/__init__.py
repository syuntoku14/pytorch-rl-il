'''
Pytorch models for batch-rl continuous control.
'''
import numpy as np
import torch
from rlil import nn


def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_bcq_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


class FC_Encoder_BCQ(nn.Module):
    def __init__(self, env, latent_dim=32, hidden1=400, hidden2=300):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(env.state_space.shape[0] +
                      env.action_space.shape[0], hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden2, latent_dim)
        self.log_var = nn.Linear(hidden2, latent_dim)

    def forward(self, states, actions):
        x = torch.cat((states.features.float(),
                       actions.features.float()), dim=1)
        x = self.head(x)
        return self.mean(x), self.log_var(x).clamp(-4, 15)


class FC_Decoder_BCQ(nn.Module):
    def __init__(self, env, latent_dim=32, hidden1=300, hidden2=400):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(env.state_space.shape[0] + latent_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, env.action_space.shape[0])
        )
        self.latent_dim = latent_dim

    def forward(self, states, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            device = next(self.parameters()).device
            z = torch.randn(states.features.size(0), self.latent_dim).to(
                device).clamp(-0.5, 0.5)

        return self.decoder(torch.cat((states.features.float(), z), dim=1))
