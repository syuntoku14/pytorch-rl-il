import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil import nn
from copy import deepcopy
from .base import Agent, LazyAgent
import os


class VaeBC(Agent):
    """
    VAE Behavioral Cloning (VAE-BC)

    VaeBC is a behavioral cloning method used in BCQ, BEAR and BRAC.
    It replaces the NN regressor in BC implementation with a VAE. 
    This code is mainly for debugging.

    Args:
        encoder (BcqEncoder): An approximation of the encoder.
        decoder (BcqDecoder): An approximation of the decoder.
        minibatch_size (int): 
            The number of experiences to sample in each training update.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 minibatch_size=100,
                 ):
        # objects
        self.encoder = encoder
        self.decoder = decoder
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.minibatch_size = minibatch_size

    def act(self, states, reward):
        # batch x num_decode x d
        vae_actions, _ = \
            self.decoder.decode_multiple(states.to(self.device), num_decode=10)
        # batch x d
        vae_actions = vae_actions.mean(1)
        return Action(vae_actions)

    def train(self):
        (states, actions, _, _, _) = self.replay_buffer.sample(
            self.minibatch_size)

        # train vae
        mean, log_var = self.encoder(
            states.to(self.device), actions.to(self.device))
        z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
        vae_actions = Action(self.decoder(states, z))
        vae_mse = mse_loss(actions.features, vae_actions.features)
        vae_kl = nn.kl_loss(mean, log_var)
        vae_loss = (vae_mse + 0.5 * vae_kl)
        self.decoder.reinforce(vae_loss)
        self.encoder.reinforce()
        self.writer.add_scalar('loss/vae/mse', vae_mse.detach())
        self.writer.add_scalar('loss/vae/kl', vae_kl.detach())
        self.writer.train_steps += 1

    def should_train(self):
        return True

    def make_lazy_agent(self, *args, **kwargs):
        decoder_model = deepcopy(self.decoder.model)
        return VaeBcLazyAgent(decoder_model.to("cpu"), *args, **kwargs)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename in ('encoder.pt'):
                self.encoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)
            if filename in ('decoder.pt'):
                self.decoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)


class VaeBcLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, decoder_model, *args, **kwargs):
        self._decoder_model = decoder_model
        super().__init__(*args, **kwargs)

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            # batch x num_decode x d
            actions, _ = \
                self._decoder_model.decode_multiple(states, num_decode=10)
            # batch x d
            self._actions = Action(actions.mean(1))
        return self._actions
