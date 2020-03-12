import unittest
import torch
import torch_testing as tt
from rlil import nn
from rlil.approximation.auto_encoder import AutoEncoder
from rlil.environments import State, Action

STATE_DIM = 5
ACTION_DIM = 2
HIDDEN_DIM = 3


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM)
        )
        self.model_decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, STATE_DIM)
        )

        model_parameters = (list(self.model_encoder.parameters()) + list(self.model_decoder.parameters()))
        optimizer = torch.optim.SGD(model_parameters, lr=0.1)
        self.ae = AutoEncoder(self.model_encoder, self.model_decoder, optimizer)
        self.criterion = nn.MSELoss()

    def test_decode(self):
        states = State(
            torch.randn(5, STATE_DIM),
        )
        enc = self.ae.encode(states.features)
        dec = self.ae.decode(enc)
        assert states.features.shape == dec.shape

    def test_reinforce(self):
        states = State(
            torch.randn(5, STATE_DIM),
        )
        enc = self.ae.encode(states.features)
        dec = self.ae.decode(enc)
        loss = self.criterion(states.features, dec)

        self.ae.zero_grad()
        for _ in range(10):
            enc = self.ae.encode(states.features)
            dec = self.ae.decode(enc)
            new_loss = self.criterion(states.features, dec)
            self.ae.reinforce(new_loss)
        assert new_loss < loss


class FC_Encoder_BCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM)
        self.log_var = nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM)

    def forward(self, states, actions_raw):
        x = torch.cat((states.features.float(), actions_raw), dim=1)
        return self.mean(x), self.log_var(x).clamp(-4, 15)


class FC_Decoder_BCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(STATE_DIM + HIDDEN_DIM, ACTION_DIM)

    def forward(self, states, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            device = next(self.parameters()).device
            z = torch.randn(states.features.size(0), HIDDEN_DIM).to(device).clamp(-0.5, 0.5)

        return self.decoder(torch.cat((states.features.float(), z), dim=1))


class TestVAE_BCQ(unittest.TestCase):
    # Test the network architecture of https://github.com/sfujim/BCQ/blob/05c07fc442a2be96f6249b966682cf065045500f/BCQ.py
    def setUp(self):
        torch.manual_seed(2)
        self.model_encoder = FC_Encoder_BCQ()
        self.model_decoder = FC_Decoder_BCQ()

        model_parameters = (list(self.model_encoder.parameters()) + list(self.model_decoder.parameters()))
        optimizer = torch.optim.SGD(model_parameters, lr=0.1)
        self.vae = AutoEncoder(self.model_encoder, self.model_decoder, optimizer)
        self.recon_criterion = nn.MSELoss()

    def test_decode(self):
        states = State(
            torch.randn(5, STATE_DIM),
        )
        actions = Action(torch.randn(5, ACTION_DIM))
        mean, log_var = self.vae.encode(states, actions.features)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
        dec = self.vae.decode(states, z)
        assert actions.features.shape == dec.shape

    def test_reinforce(self):
        states = State(
            torch.randn(5, STATE_DIM),
        )
        actions = Action(torch.randn(5, ACTION_DIM))
        mean, log_var = self.vae.encode(states, actions.features)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
        dec = self.vae.decode(states, z)
        loss = self.recon_criterion(actions.features, dec) + nn.kl_loss(mean, log_var)

        for _ in range(10):
            mean, log_var = self.vae.encode(states, actions.features)
            z = mean + log_var.exp() * torch.randn_like(log_var)
            dec = self.vae.decode(states, z)
            new_loss = self.recon_criterion(actions.features, dec) + nn.kl_loss(mean, log_var)
            self.vae.reinforce(new_loss)
        assert new_loss < loss


if __name__ == '__main__':
    unittest.main()
