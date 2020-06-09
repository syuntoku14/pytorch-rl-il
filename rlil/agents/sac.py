import torch
import os
from copy import deepcopy
from rlil.environments import Action
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex, get_n_step)
from rlil.memory import ExperienceReplayBuffer
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
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
        self.n_step, _ = get_n_step()
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
            samples = Samples(self._states, self._actions, reward, states)
            self.replay_buffer.store(samples)
        self._states = states
        self._actions = Action(self.policy.no_grad(
            states.to(self.device))[0]).to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            # sample from replay buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)

            # Target actions come from *current* policy
            _actions, _log_probs = self.policy.no_grad(states)
            # compute targets for Q and V
            q_targets = rewards + (self.discount_factor**self.n_step) * \
                self.v.target(next_states)
            v_targets = torch.min(
                self.q_1.target(states, Action(_actions)),
                self.q_2.target(states, Action(_actions)),
            ) - self.temperature * _log_probs

            # update Q and V-functions
            q_1_values = self.q_1(states, actions)
            self.q_1.reinforce(
                weighted_mse_loss(q_1_values, q_targets, weights))
            q_2_values = self.q_2(states, actions)
            self.q_2.reinforce(
                weighted_mse_loss(q_2_values, q_targets, weights))
            self.v.reinforce(weighted_mse_loss(
                self.v(states), v_targets, weights))

            # update priorities
            td_errors = (q_targets - q_1_values).abs()
            self.replay_buffer.update_priorities(indexes, td_errors.cpu())

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
            self.writer.add_scalar('loss/td_error', td_errors.mean())

            self.writer.train_steps += 1

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        q_model = deepcopy(self.q_1.model)
        v_target_model = deepcopy(self.v._target._target)
        return SACLazyAgent(policy_model.to("cpu"),
                            q_model=q_model.to("cpu"),
                            v_target_model=v_target_model.to("cpu"),
                            discount_factor=self.discount_factor,
                            use_apex=use_apex,
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

    def __init__(self, policy_model,
                 q_model, v_target_model, discount_factor,
                 use_apex, *args, **kwargs):
        self._policy_model = policy_model
        self._q_model = q_model
        self._v_target_model = v_target_model
        self._discount_factor = discount_factor
        self._use_apex = use_apex
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()

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

    def compute_priorities(self, samples):
        if self._use_apex:
            targets = samples.rewards + self._discount_factor * \
                self._v_target_model(samples.next_states)
            q_values = self._q_model(samples.states, samples.actions)
            td_errors = (targets - q_values).abs()
            return td_errors
        else:
            return None
