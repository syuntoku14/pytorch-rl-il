import torch
import os
from copy import deepcopy
from torch.nn.functional import mse_loss
from rlil.environments import Action
from rlil.initializer import get_writer, get_device, get_replay_buffer
from rlil.memory import ExperienceReplayBuffer
from .base import Agent, LazyAgent


class SAC(Agent):
    """
    Soft Actor-Critic (SAC).
    SAC is a proposed improvement to DDPG that replaces the standard
    mean-squared Bellman error (MSBE) objective with a "maximum entropy"
    objective that impoves exploration. It also uses a few other tricks,
    such as the "Clipped Double-Q Learning" trick introduced by TD3.
    This implementation uses automatic temperature adjustment to replace the
    difficult to set temperature parameter with a more easily tuned
    entropy target parameter.
    https://arxiv.org/abs/1801.01290

    Args:
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        q1 (QContinuous): An Approximation of the continuous action Q-function.
        q2 (QContinuous): An Approximation of the continuous action Q-function.
        v (VNetwork): An Approximation of the state-value function.
        discount_factor (float): Discount factor for future rewards.
        entropy_target (float): The desired entropy of the policy. Usually -env.action_space.shape[0]
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        temperature_initial (float): The initial temperature used in the maximum entropy objective.
    """

    def __init__(self,
                 policy,
                 q_1,
                 q_2,
                 v,
                 discount_factor=0.99,
                 entropy_target=-2.,
                 lr_temperature=1e-4,
                 minibatch_size=32,
                 replay_start_size=5000,
                 temperature_initial=0.1,
                 ):
        # objects
        self.policy = policy
        self.v = v
        self.q_1 = q_1
        self.q_2 = q_2
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_target = entropy_target
        self.lr_temperature = lr_temperature
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.temperature = temperature_initial
        # private
        self._states = None
        self._actions = None

    def act(self, states, reward=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        self._actions = Action(self.policy.eval(
            states.to(self.device))[0]).to("cpu")
        return self._actions

    def train(self):
        if self._should_train():
            # sample from replay buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)

            # compute targets for Q and V
            _actions, _log_probs = self.policy.eval(states)
            q_targets = rewards + self.discount_factor * \
                self.v.target(next_states)
            v_targets = torch.min(
                self.q_1.target(states, Action(_actions)),
                self.q_2.target(states, Action(_actions)),
            ) - self.temperature * _log_probs

            # update Q and V-functions
            self.q_1.reinforce(
                mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(
                mse_loss(self.q_2(states, actions), q_targets))
            self.v.reinforce(mse_loss(self.v(states), v_targets))

            # update policy
            _actions2, _log_probs2 = self.policy(states)
            loss = (-self.q_1(states, Action(_actions2)) +
                    self.temperature * _log_probs2).mean()
            self.policy.reinforce(loss)

            # adjust temperature
            temperature_grad = (_log_probs + self.entropy_target).mean()
            self.temperature += self.lr_temperature * temperature_grad.detach()

            # additional debugging info
            self.writer.add_scalar('loss/entropy', -_log_probs.mean())
            self.writer.add_scalar('loss/v_mean', v_targets.mean())
            self.writer.add_scalar('loss/r_mean', rewards.mean())
            self.writer.add_scalar('loss/temperature_grad', temperature_grad)
            self.writer.add_scalar('loss/temperature', self.temperature)

            self.writer.train_steps += 1

    def _should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        model = deepcopy(self.policy.model)
        return SACLazyAgent(model.to("cpu"),
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


class SACLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, *args, **kwargs):
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            if self._evaluation:
                outputs = self._policy_model(states, return_mean=True)
            else:
                outputs = self._policy_model(states)[0]
            self._actions = Action(outputs).to("cpu")
        return self._actions
