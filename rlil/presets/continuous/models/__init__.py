'''
Pytorch models for continuous control.

All models assume that a feature representing the
current timestep is used in addition to the features
received from the environment.
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


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.ReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


class FC_Encoder_BCQ(nn.Module):
    def __init__(self, env, latent_dim=32, hidden1=400, hidden2=300):
        self.head = nn.Sequential(
            nn.Linear(env.state_space.shape[0] +
                      env.action_space.shape[0], hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden2, latent_dim)
        self.log_var = nn.Linear(hidden2, latent_dim)

    def forward(self, states, actions_raw):
        x = torch.cat((states.features.float(), actions_raw), dim=1)
        x = self.head(x)
        return self.mean(x), self.log_var(x).clamp(-4, 15)


class FC_Decoder_BCQ(nn.RLNetwork):
    def __init__(self, env, latent_dim=32, hidden1=300, hidden2=400):
        self.decoder = nn.Sequential(
            nn.Linear(env.state_space.shape[0] + latent_dim),
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
