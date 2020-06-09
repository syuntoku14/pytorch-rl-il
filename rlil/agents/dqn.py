import torch
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex, get_n_step)
from rlil.policies import GreedyPolicy
from rlil.environments import Action
from .base import Agent, LazyAgent
from copy import deepcopy
import os


class DQN(Agent):
    '''
    Deep Q-Network (DQN).
    DQN was one of the original deep reinforcement learning algorithms.
    It extends the ideas behind Q-learning to work well with modern convolution networks.
    The core innovation is the use of a replay buffer, which allows the use of batch-style
    updates with decorrelated samples. It also uses a "target" network in order to
    improve the stability of updates.
    https://www.nature.com/articles/nature14236

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        epsilon (rlil.utils.scheduler.LinearScheduler): 
            For epsilon greedy exploration.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        n_actions (int): The number of available actions.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
    '''

    def __init__(self,
                 q,
                 policy,
                 epsilon,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.epsilon = epsilon
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        self.loss = weighted_mse_loss
        # hyperparameters
        self.n_step, _ = get_n_step()
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        # private
        self._states = None
        self._actions = None

    def act(self, states, rewards=None):
        if rewards is not None:
            samples = Samples(self._states, self._actions, rewards, states)
            self.replay_buffer.store(samples)
        self._states = states
        self._actions = Action(
            self.policy.no_grad(states.to(self.device))).to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            targets = rewards + (self.discount_factor**self.n_step) * \
                torch.max(self.q.target(next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets, weights)
            # backward pass
            self.q.reinforce(loss)
            # update epsilon greedy
            self.epsilon.update()
            self.policy.set_epsilon(self.epsilon.get())
            self.writer.train_steps += 1

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename in ('q.pt'):
                self.q.model = torch.load(os.path.join(dirname, filename),
                                          map_location=self.device)

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        q_model = deepcopy(self.q.model).to("cpu")
        q_target_model = deepcopy(self.q._target._target).to("cpu")
        num_actions = Action.action_space().n
        policy = GreedyPolicy(q_model, num_actions, epsilon=self.epsilon.get())
        return DQNLazyAgent(q_model=q_model,
                            q_target_model=q_target_model,
                            policy=policy,
                            discount_factor=self.discount_factor,
                            use_apex=use_apex(),
                            evaluation=evaluation,
                            store_samples=store_samples)


class DQNLazyAgent(LazyAgent):
    def __init__(self, q_model, q_target_model, policy,
                 discount_factor, use_apex, *args, **kwargs):
        self._q_model = q_model
        self._q_target_model = q_target_model
        self._policy = policy
        self._discount_factor = discount_factor
        self._use_apex = use_apex
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._q_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        if self._evaluation:
            self._actions = Action(self._policy.eval(states))
        else:
            self._actions = Action(self._policy.no_grad(states))
        return self._actions

    def compute_priorities(self, samples):
        if self._use_apex:
            q_values = self._q_model(samples.states, samples.actions)
            targets = samples.rewards + self._discount_factor * \
                torch.max(self._q_target_model(samples.next_states), dim=1)[0]
            priorities = (targets - q_values).abs()
            return priorities
        else:
            return None
