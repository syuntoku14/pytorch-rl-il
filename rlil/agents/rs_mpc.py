import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import mse_loss
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil import nn
from copy import deepcopy
from .base import Agent, LazyAgent
import os


class RsMPC(Agent):
    """
    Random shooting MPC (RsMPC).
    This implementation is based on: https://arxiv.org/abs/1708.02596.
    The random shooting method generates random actions at each MPC iteration.
    It chooses the best from the candidates based on the reward function 
    and the dynamics model.

    Args:
        dynamics (rlil.approximation.Dynamics): 
            An Approximation of a dynamics model.
        reward_fn (rlil.environments.reward_fn): Reward function for mpc
        horizon (int): Control horizon.
        num_samples (int): Number of action samples for random shooting.
        minibatch_size (int): 
            The number of experiences to sample in each training update.
        replay_start_size (int): 
            Number of experiences in replay buffer when training begins.
    """

    def __init__(self,
                 dynamics,
                 reward_fn,
                 horizon,
                 num_samples,
                 minibatch_size=32,
                 replay_start_size=5000,
                 ):
        # objects
        self.dynamics = dynamics
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        self._reward_fn = reward_fn
        self._action_space = Action.action_space()
        self._action_uniform = Uniform(
            low=torch.tensor(self._action_space.low,
                             dtype=torch.float32, device=self.device),
            high=torch.tensor(
                self._action_space.high, dtype=torch.float32, device=self.device),
        )
        # hyperparameters
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self._num_samples = num_samples
        self._horizon = horizon
        # private
        self._states = None
        self._actions = None

    def _make_random_actions(self, num_samples):
        actions = self._action_uniform.sample([num_samples])
        return Action(actions).to(self.device)

    def _mpc(self, state):
        init_actions = self._make_random_actions(self._num_samples)
        total_rewards = torch.zeros(self._num_samples, device=self.device)
        state = State(state.features.repeat(
            self._num_samples, 1).to(self.device))
        for i in range(self._horizon):
            if i == 0:
                actions = init_actions
            else:
                actions = self._make_random_actions(self._num_samples)
            next_state = self.dynamics(state, actions)
            rewards = self._reward_fn(state, next_state, actions)
            total_rewards += rewards
            state = next_state
        idx = total_rewards.argmax()
        return init_actions[idx.item()]

    def act(self, states, reward=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        if self.should_train():
            actions = self._make_random_actions(len(states))
        else:
            actions = Action.from_list([self._mpc(state) for state in states])
        self._actions = actions.to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            (states, actions, _, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)
            predictions = self.dynamics(states, actions)
            loss = mse_loss(next_states.features, predictions.features)
            self.dynamics.reinforce(loss)
            self.writer.train_steps += 1

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, *args, **kwargs):
        model = deepcopy(self.dynamics.model)
        action_uniform = Uniform(
            low=torch.tensor(self._action_space.low,
                             dtype=torch.float32, device="cpu"),
            high=torch.tensor(self._action_space.high,
                              dtype=torch.float32, device="cpu"))

        return RsMpcLazyAgent(
            model.to("cpu"),
            self._reward_fn,
            action_uniform,
            self._horizon,
            self._num_samples,
            self.should_train(),
            *args, **kwargs)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'dynamics.pt':
                self.dynamics.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)


class RsMpcLazyAgent(RsMPC, LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self,
                 dynamics_model,
                 reward_fn,
                 action_uniform,
                 horizon,
                 num_samples,
                 should_train,
                 *args, **kwargs):
        self.dynamics = dynamics_model
        self._reward_fn = reward_fn
        self._action_space = Action.action_space()
        self._action_uniform = action_uniform
        self._num_samples = num_samples
        self._horizon = horizon
        self._should_train = should_train
        self.device = "cpu"
        LazyAgent.__init__(self, *args, **kwargs)
        if self._evaluation:
            self.dynamics.eval()

    def act(self, states, reward):
        LazyAgent.act(self, states, reward)
        self._states = states
        with torch.no_grad():
            if self._evaluation or self._should_train:
                actions = Action.from_list(
                    [self._mpc(state) for state in states])
            else:
                actions = self._make_random_actions(len(states))
            self._actions = actions.to("cpu")
        return self._actions
