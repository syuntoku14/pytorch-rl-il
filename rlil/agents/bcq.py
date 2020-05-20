import os
import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from copy import deepcopy
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_writer, get_device, get_replay_buffer
from rlil import nn
from .base import Agent, LazyAgent


class BCQ(Agent):
    """
    Batch-Constrained Q-learning (BCQ)

    BCQ is an algorithm to train an agent from a fixed batch.
    Traditional off-policy algorithms such as DQN and DDPG fail to train an agent from a fixed batch
    due to extraporation error. Extraporation error causes overestimation of the q values for state-action
    pairs that fall outside of the distribution of the fixed batch.
    BCQ attempts to eliminate extrapolation error by constraining the agent's actions to the data
    distribution of the batch. 
    https://arxiv.org/abs/1812.02900

    This implementation is based on: https://github.com/sfujim/BCQ.

    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        encoder (BcqEncoder): An approximation of the encoder.
        decoder (BcqDecoder): An approximation of the decoder.
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        lambda_q (float): Weight for soft clipped double q-learning
    """

    def __init__(self,
                 q_1,
                 q_2,
                 encoder,
                 decoder,
                 policy,
                 discount_factor=0.99,
                 lambda_q=0.75,
                 minibatch_size=32,
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
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.lambda_q = lambda_q

    def act(self, states, rewards):
        states = State(states.features.repeat(100, 1).to(self.device))
        vae_actions = Action(self.decoder(states))
        policy_actions = Action(self.policy.no_grad(states, vae_actions))
        q_1 = self.q_1(states, policy_actions)
        ind = q_1.argmax(0).item()
        return policy_actions[ind].to("cpu")

    def train(self):
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
            next_vae_actions = Action(self.decoder(next_states_10))
            next_actions = Action(
                self.policy.target(next_states_10, next_vae_actions))
            q_1_targets = self.q_1.target(next_states_10, next_actions)
            q_2_targets = self.q_2.target(next_states_10, next_actions)

            # Soft Clipped Double Q-learning
            q_targets = self.lambda_q * torch.min(q_1_targets, q_2_targets) \
                + (1. - self.lambda_q) * torch.max(q_1_targets, q_2_targets)
            # Take max over each action sampled from the VAE
            q_targets = q_targets.reshape(
                self.minibatch_size, -1).max(1)[0].reshape(-1, 1)
            q_targets = rewards.reshape(-1, 1) + \
                self.discount_factor * q_targets * next_states.mask.float().reshape(-1, 1)

        self.q_1.reinforce(
            mse_loss(self.q_1(states, actions).reshape(-1, 1), q_targets))
        self.q_2.reinforce(
            mse_loss(self.q_2(states, actions).reshape(-1, 1), q_targets))

        # train policy
        vae_actions = Action(self.decoder(states))
        sampled_actions = Action(self.policy(states, vae_actions))
        loss = -self.q_1(states, sampled_actions).mean()
        self.policy.reinforce(loss)

        self.writer.train_steps += 1

    def should_train(self):
        return True

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        q_1_model = deepcopy(self.q_1.model)
        decoder_model = deepcopy(self.decoder.model)
        return BcqLazyAgent(policy_model.to("cpu"),
                            q_1_model.to("cpu"),
                            decoder_model.to("cpu"),
                            evaluation=evaluation,
                            store_samples=store_samples)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)
            if filename in ('q_1.pt'):
                self.q_1.model = torch.load(os.path.join(dirname, filename),
                                            map_location=self.device)
            if filename in ('q_2.pt'):
                self.q_1.model = torch.load(os.path.join(dirname, filename),
                                            map_location=self.device)
            if filename in ('encoder.pt'):
                self.encoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)
            if filename in ('decoder.pt'):
                self.decoder.model = torch.load(os.path.join(dirname, filename),
                                                map_location=self.device)


class BcqLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self,
                 policy_model,
                 q_1_model,
                 decoder_model,
                 *args, **kwargs):
        self._policy_model = policy_model
        self._q_1_model = q_1_model
        self._decoder_model = decoder_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()
            self._q_1_model.eval()
            self._decoder_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            states = State(torch.tensor(
                states.features.repeat(100, 1),
                dtype=torch.float32))
            vae_actions = Action(self._decoder_model(states))
            policy_actions = self._policy_model(states, vae_actions)
            policy_actions = Action(policy_actions)
            q_1 = self._q_1_model(states, policy_actions)
            ind = q_1.argmax(0).item()
            actions = policy_actions[ind]
        self._actions = actions
        return self._actions
