import os
import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from copy import deepcopy
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_writer, get_device, get_replay_buffer
from rlil import nn
from .base import Agent, LazyAgent


class BEAR(Agent):
    """
    Bootstrapping error accumulation reduction (BEAR)

    BEAR is an algorithm to train an agent from a fixed batch.
    Unlike BCQ, BEAR doesn't constraint the distribution of the 
    learned policy to be close to the behavior policy.
    Instead, BEAR restricts the policy to ensure that the actions of 
    the policy lies in the support of the training distribution.
    https://arxiv.org/abs/1906.00949

    This implementation is based on: https://github.com/aviralkumar2907/BEAR

    Args:
        qs (EnsembleQContinuous): An Approximation of the continuous action Q-functions.
        encoder (BcqEncoder): An approximation of the encoder.
        decoder (BcqDecoder): An approximation of the decoder.
        policy (SoftDeterministicPolicy): An Approximation of a soft deterministic policy.
        kernel_type: ("gaussian"|"laplacian") Kernel type used for computation of MMD
        num_samples_match (int): Number of samples used for computing sampled MMD
        mmd_sigma (float): Parameter for computation of MMD.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        lambda_q (float): Weight for soft clipped double q-learning
        _lambda (float): Weight for actor loss with mmd_loss
    """

    def __init__(self,
                 qs,
                 encoder,
                 decoder,
                 policy,
                 kernel_type="laplacian",
                 num_samples_match=5,
                 mmd_sigma=10.0,
                 discount_factor=0.99,
                 lambda_q=0.75,
                 _lambda=0.4,
                 delta_conf=0.1,
                 minibatch_size=32,
                 ):
        # objects
        self.qs = qs
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        # hyperparameters
        self.kernel_type = kernel_type
        self.mmd_sigma = mmd_sigma
        self.num_samples_match = num_samples_match
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.lambda_q = lambda_q
        self._lambda = _lambda
        self.delta_conf = delta_conf
        # lagrange multipliers for maintaining support matching at all times
        self.log_lagrange2 = torch.randn(
            (), requires_grad=True, device=self.device)
        self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2, ], lr=1e-3)
        # private
        self._train_count = 0

    def act(self, states, rewards):
        with torch.no_grad():
            states = State(states.features.repeat(10, 1).to(self.device))
            policy_actions = Action(self.policy.no_grad(states)[0])
            q1_values = self.qs.q1(states, policy_actions)
            ind = q1_values.argmax(0).item()
            return policy_actions[ind].to("cpu")

    def train(self):
        self._train_count += 1

        # sample transitions from buffer
        (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
            self.minibatch_size)

        # Train the Behaviour cloning policy to be able to
        # take more than 1 sample for MMD
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

        # train critic
        with torch.no_grad():
            # Duplicate next state 10 times
            next_states_10 = State(torch.repeat_interleave(
                next_states.features, 10, 0).to(self.device))

            # Compute value of perturbed actions sampled from the VAE
            next_vae_actions = Action(self.decoder(next_states_10))
            next_actions = Action(
                self.policy.target(next_states_10, next_vae_actions))
            # (batch x 10) x num_q
            qs_targets = self.qs.target(next_states_10, next_actions)

            # Soft Clipped Double Q-learning
            # (batch x 10) x 1
            q_targets = self.lambda_q * qs_targets.min(1)[0] \
                + (1. - self.lambda_q) * qs_targets.max(1)[0]
            # Take max over each action sampled from the VAE
            # batch x 1
            q_targets = q_targets.reshape(
                self.minibatch_size, -1).max(1)[0].reshape(-1, 1)
            q_targets = rewards.reshape(-1, 1) + \
                self.discount_factor * q_targets * \
                next_states.mask.float().reshape(-1, 1)

        current_qs = self.qs(states, actions)  # batch x num_q
        q_targets = torch.repeat_interleave(q_targets, current_qs.shape[1], 1)
        q_loss = mse_loss(current_qs, q_targets)
        self.qs.reinforce(q_loss)

        # train policy
        # batch x num_samples_match x d
        vae_actions, raw_vae_actions = \
            self.decoder.decode_multiple(states, self.num_samples_match)
        sampled_actions, raw_sampled_actions = \
            self.policy.sample_multiple(states, self.num_samples_match)

        if self.kernel_type == 'gaussian':
            mmd_loss = nn.mmd_loss_gaussian(
                raw_vae_actions, raw_sampled_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = nn.mmd_loss_laplacian(
                raw_vae_actions, raw_sampled_actions, sigma=self.mmd_sigma)

        action_divergence = ((vae_actions - sampled_actions)**2).sum(-1)
        raw_action_divergence = (
            (raw_vae_actions - raw_sampled_actions)**2).sum(-1)

        # Update through TD3 style
        # (batch x num_samples_match) x d
        repeated_states = torch.repeat_interleave(
            states.features.unsqueeze(1),
            self.num_samples_match, 1).view(-1, states.features.shape[1])
        repeated_actions = \
            sampled_actions.contiguous().view(-1, sampled_actions.shape[2])
        # (batch x num_samples_match) x num_q
        critic_qs = self.qs(State(repeated_states), Action(repeated_actions))
        # batch x num_samples_match x num_q
        critic_qs = \
            critic_qs.view(-1, self.num_samples_match, critic_qs.shape[1])
        critic_qs = critic_qs.mean(1)  # batch x num_q
        std_q = torch.std(critic_qs, dim=-1, keepdim=False,
                          unbiased=False)  # batch
        critic_qs = critic_qs.min(1)[0]  # batch

        # Do support matching with a warmstart which happens to be reasonable
        # around epoch 20 during training
        if self._train_count >= 20:
            actor_loss = (-critic_qs + self._lambda *
                          (np.sqrt((1 - self.delta_conf) / self.delta_conf)) *
                          std_q + self.log_lagrange2.exp() * mmd_loss).mean()
        else:
            actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()

        std_loss = self._lambda * (np.sqrt((1 - self.delta_conf) /
                                           self.delta_conf)) * std_q.detach().mean()
        actor_loss.backward(retain_graph=True)
        self.policy.reinforce(actor_loss, backward=False)

        # update lagrange multipliers
        thresh = 0.05
        lagrange_loss = (-critic_qs.detach() + self._lambda *
                         (np.sqrt((1 - self.delta_conf) / self.delta_conf)) *
                         std_q.detach() + self.log_lagrange2.exp() *
                         (mmd_loss.detach() - thresh)).mean()

        self.lagrange2_opt.zero_grad()
        (-lagrange_loss).backward()
        self.lagrange2_opt.step()
        self.log_lagrange2.data.clamp_(min=-5.0, max=10.0)

        self.writer.add_scalar('loss/mmd', mmd_loss.detach().mean())
        self.writer.add_scalar('loss/actor', actor_loss.detach())
        self.writer.add_scalar('loss/qs', q_loss.detach())
        self.writer.add_scalar('loss/std', std_loss.detach())
        self.writer.add_scalar('loss/lagrange2', lagrange_loss.detach())
        self.writer.add_scalar('critic_qs', critic_qs.detach().mean())
        self.writer.add_scalar('std_q', std_q.detach().mean())
        self.writer.add_scalar('lagrange2', self.log_lagrange2.exp().detach())

        self.writer.train_steps += 1

    def should_train(self):
        return True

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        qs_model = deepcopy(self.qs.model)
        return BearLazyAgent(policy_model.to("cpu"),
                             qs_model.to("cpu"),
                             evaluation=evaluation,
                             store_samples=store_samples)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)
            if filename in ('qs.pt'):
                self.qs.model = torch.load(os.path.join(dirname, filename),
                                           map_location=self.device)
            if filename in ('encoder.pt'):
                self.encoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)
            if filename in ('decoder.pt'):
                self.decoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)


class BearLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self,
                 policy_model,
                 qs_model,
                 *args, **kwargs):
        self._policy_model = policy_model
        self._qs_model = qs_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()
            self._qs_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            states = State(states.features.repeat(10, 1))
            policy_actions = Action(self._policy_model(states)[0])
            q1_values = self._qs_model.q1(states, policy_actions)
            ind = q1_values.argmax(0).item()
            self._actions = policy_actions[ind].to("cpu")
            return self._actions
