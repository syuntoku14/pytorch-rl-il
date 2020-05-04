import pytest
import torch
import torch_testing as tt
from torch.nn.functional import mse_loss
from rlil import nn
from rlil.approximation.bcq_auto_encoder import BcqEncoder, BcqDecoder
from rlil.environments import State, Action, GymEnvironment
from rlil.presets.continuous.models import fc_bcq_encoder, fc_bcq_decoder
import numpy as np


# Test the network architecture of
# https://github.com/sfujim/BCQ/blob/05c07fc442a2be96f6249b966682cf065045500f/BCQ.py
@pytest.fixture
def setUp():
    env = GymEnvironment('LunarLanderContinuous-v2')
    Action.set_action_space(env.action_space)
    latent_dim = 32
    encoder_model = fc_bcq_encoder(env, latent_dim=latent_dim)
    decoder_model = fc_bcq_decoder(env, latent_dim=latent_dim)

    encoder_optimizer = torch.optim.SGD(encoder_model.parameters(), lr=0.1)
    decoder_optimizer = torch.optim.SGD(decoder_model.parameters(), lr=0.1)
    encoder = BcqEncoder(model=encoder_model,
                         latent_dim=latent_dim,
                         optimizer=encoder_optimizer)
    decoder = BcqDecoder(model=decoder_model,
                         latent_dim=latent_dim,
                         space=env.action_space,
                         optimizer=decoder_optimizer)

    sample_states = env.reset()
    sample_actions = Action(
        torch.tensor(env.action_space.sample()).unsqueeze(0))

    yield encoder, decoder, sample_states, sample_actions


def test_decode(setUp):
    encoder, decoder, states, actions = setUp
    mean, log_var = encoder(states, actions)
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    assert actions.features.shape == dec.shape


def test_reinforce(setUp):
    encoder, decoder, states, actions = setUp
    mean, log_var = encoder(states, actions)
    # reinforce mse
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    loss = mse_loss(actions.features, dec)

    for _ in range(10):
        mean, log_var = encoder(states, actions)
        z = mean + log_var.exp() * torch.randn_like(log_var)
        dec = decoder(states, z)
        new_loss = mse_loss(actions.features, dec)
        decoder.reinforce(new_loss)
        encoder.reinforce()
    assert new_loss < loss

    # reinforce mse
    z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
    dec = decoder(states, z)
    loss = nn.kl_loss(mean, log_var)

    for _ in range(10):
        mean, log_var = encoder(states, actions)
        z = mean + log_var.exp() * torch.randn_like(log_var)
        dec = decoder(states, z)
        new_loss = nn.kl_loss(mean, log_var)
        decoder.reinforce(new_loss)
        encoder.reinforce()
    assert new_loss < loss

