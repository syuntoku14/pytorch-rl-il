import os
import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from copy import deepcopy
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_writer, get_device, get_replay_buffer
from rlil import nn
from tqdm import tqdm
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
    divergence work well.
    3 Whether to learn lagrange multipliers alpha adaptively. 
     The paper says that a fixed alpha is superior than an adaptive alpha.
    4 Whether to use a value penalty in the Q function objective or just use policy regularization.
    The paper says that the performance imploves slightly when using a value
     penalty in the Q function objective.

    Then, this implementation uses following tricks:
    1. double Q learning like TD3, 2. use KL divergence,
    3. fixed alpha, 4. Q function objective penalty.

    The behavior policy is estimated using simple bc with SoftDeterministicPolicy.
    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        policy (SoftDeterministicPolicy): An Approximation of a soft deterministic policy.
        behavior_policy (SoftDeterministicPolicy): An approximation of the behavior policy.
        bc_iters (int): Number of training steps for behavior cloning.
        alpha (float): Value of lagrange multipliers. Trick 3.
        n_div_samples (int): Number of samples used for computing divergence.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
    """

    def __init__(self,
                 q_1,
                 q_2,
                 policy,
                 behavior_policy,
                 bc_iters=5000,
                 alpha=0.1,
                 n_div_samples=4,
                 discount_factor=0.99,
                 minibatch_size=100,
                 ):
        # objects
        self.q_1 = q_1
        self.q_2 = q_2
        self.policy = policy
        self.behavior_policy = behavior_policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        # hyperparameters
        self.alpha = alpha
        self.bc_iters = bc_iters
        self.n_div_samples = n_div_samples
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._train_count = 0

    def act(self, states, reward=None):
        actions = Action(self.policy.no_grad(
            states.to(self.device))[0]).to("cpu")
        return actions

    def train(self):
        if self._train_count == 0:
            self.train_bc()

        self._train_count += 1

        # sample transitions from buffer
        (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
            self.minibatch_size)

        # Trick 2: KL divergence regularization
        policy_mean, policy_logvar = self.policy.mean_logvar(states)
        behavior_mean, behavior_logvar = self.behavior_policy.mean_logvar(
            states)
        kl = nn.kl_gaussian(policy_mean, policy_logvar,
                            behavior_mean, behavior_logvar)

        # Trick 1: clipped double Q learning
        next_actions, _ = self.policy.target(next_states)
        q_values = torch.min(self.q_1.target(next_states, Action(next_actions)),
                             self.q_2.target(next_states, Action(next_actions)))

        # Trick 4: Q target with divergence penalty
        q_targets = rewards + self.discount_factor * \
            (q_values - self.alpha * kl.detach())

        self.q_1.reinforce(
            mse_loss(self.q_1(states, actions), q_targets))
        self.q_2.reinforce(
            mse_loss(self.q_2(states, actions), q_targets))

        # Update policy with a warmstart
        policy_loss = self.alpha * kl
        if self._train_count >= 5000:
            policy_actions, _ = self.policy(states)
            policy_loss -= self.q_1(states, Action(policy_actions))
        self.policy.reinforce(policy_loss.mean())

        self.writer.add_scalar('q_targets/mean', q_targets.detach().mean())
        self.writer.add_scalar('loss/kl', kl.detach().mean())
        self.writer.train_steps += 1

    def train_bc(self):
        for _ in tqdm(range(self.bc_iters)):
            (states, actions, _, _, _) = self.replay_buffer.sample(
                self.minibatch_size)
            bc_actions = Action(self.behavior_policy(states)[0])
            loss = mse_loss(bc_actions.features, actions.features)
            self.behavior_policy.reinforce(loss)

        # freeze behavior policy
        for param in self.behavior_policy.model.parameters():
            param.requires_grad = False

    def should_train(self):
        return True

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        return BracLazyAgent(policy_model.to("cpu"),
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
            if filename in ('behavior_policy.pt'):
                self.behavior_policy.model = torch.load(os.path.join(dirname, filename),
                                                        map_location=self.device)


class BracLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, *args, **kwargs):
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            outputs = self._policy_model(states, return_mean=True)
            self._actions = Action(outputs).to("cpu")
        return self._actions
