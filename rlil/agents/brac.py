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


class BRAC(Agent):
    """
    Behavior Regularized Actor Critic (BRAC)

    BRAC is a general framework for evaluating proposed offline RL methods
    as well as a number of simple baselines across a variety of offline 
    continuous control tasks.
    https://arxiv.org/abs/1911.11361

    There are four main options for the behavior regularized actor critic framework: 
    1 How to estimate the Q value (e.g. the BEAR uses weighted mixture of Q-values).
     According to the BRAC paper, double Q learning like TD3 works well.
    2 Which divergence to use. The paper says that KL divergence works well.
    3 Whether to learn lagrange multipliers alpha adaptively. 
     The paper says that a fixed alpha is superior than an adaptive alpha.
    4 Whether to use a value penalty in the Q function objective or just use policy regularization.
    The paper says that the performance imploves slightly when using a value
     penalty in the Q function objective.

    Then, this implementation uses following tricks:
    1. double Q learning like TD3, 2. KL divergence using behavior policy cloning with VAE, 
    3. fixed alpha == 1.0, 4. Q function objective penalty.

    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        encoder (BcqEncoder): An approximation of the encoder.
        decoder (BcqDecoder): An approximation of the decoder.
        policy (SoftDeterministicPolicy): An Approximation of a soft deterministic policy.
        alpha (float): Value of lagrange multipliers.
        num_samples_match (int): Number of samples used for computing divergence.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
    """

    def __init__(self,
                 q_1,
                 q_2,
                 encoder,
                 decoder,
                 policy,
                 alpha=1.0,
                 num_samples_match=5,
                 discount_factor=0.99,
                 minibatch_size=100,
                 ):
        # objects
        self.q_1 = q_1
        self.q_2 = q_2
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        # hyperparameters
        self.alpha = alpha
        self.num_samples_match = num_samples_match
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
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

        # train critic
        with torch.no_grad():
            # Duplicate next state 10 times
            next_states_10 = State(torch.repeat_interleave(
                next_states.features, 10, 0).to(self.device))

            # Compute value of perturbed actions sampled from the VAE
            next_vae_actions_10 = Action(self.decoder(next_states_10))
            next_actions_10 = Action(
                self.policy.target(next_states_10, next_vae_actions_10))
            q_1_targets = self.q_1.target(next_states_10, next_actions)
            q_2_targets = self.q_2.target(next_states_10, next_actions)
            q_targets = torch.min(q_1_targets, q_2_targets)
            # Take max over each action sampled from the VAE
            q_targets = q_targets.reshape(
                self.minibatch_size, -1).max(1)[0].reshape(-1, 1)
            q_targets = rewards.reshape(-1, 1) + \
                self.discount_factor * q_targets * \
                next_states.mask.float().reshape(-1, 1)

        self.q_1.reinforce(
            mse_loss(self.q_1(states, actions).reshape(-1, 1), q_targets))
        self.q_2.reinforce(
            mse_loss(self.q_2(states, actions).reshape(-1, 1), q_targets))

        # train policy
        # batch x num_samples_match x d
        vae_actions, raw_vae_actions = \
            self.decoder.decode_multiple(states, self.num_samples_match)
        actor_actions, raw_actor_actions = \
            self.policy.sample_multiple(states, self.num_samples_match)

        if self.kernel_type == 'gaussian':
            mmd_loss = nn.mmd_loss_gaussian(
                raw_vae_actions, raw_actor_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = nn.mmd_loss_laplacian(
                raw_vae_actions, raw_actor_actions, sigma=self.mmd_sigma)

        # Update through TD3 style
        # (batch x num_samples_match) x d
        repeated_states = torch.repeat_interleave(
            states.features.unsqueeze(1),
            self.num_samples_match, 1).view(-1, states.features.shape[1])
        repeated_actions = \
            actor_actions.contiguous().view(-1, actor_actions.shape[2])
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
                          std_q + self.log_lagrange2.exp().detach() * mmd_loss).mean()
        else:
            actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()

        std_loss = self._lambda * (np.sqrt((1 - self.delta_conf) /
                                           self.delta_conf)) * std_q.detach().mean()
        self.policy.reinforce(actor_loss)

        # update lagrange multipliers
        thresh = 0.05
        lagrange_loss = (self.log_lagrange2.exp() *
                         (mmd_loss - thresh).detach()).mean()

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
