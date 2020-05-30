import torch
import os
from copy import deepcopy
from torch.distributions.normal import Normal
from rlil.environments import Action
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex)
from rlil.memory import ExperienceReplayBuffer
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
from .base import Agent, LazyAgent
from .ddpg import DDPGLazyAgent


class TD3(Agent):
    """
    Twin Dueling DDPG(TD3). 
    The following description is cited from openai-spining up(https://spinningup.openai.com/en/latest/algorithms/td3.html)

    While DDPG can achieve great performance sometimes, it is frequently 
    brittle with respect to hyperparameters and other kinds of tuning. 
    A common failure mode for DDPG is that the learned Q-function begins 
    to dramatically overestimate Q-values, which then leads to the policy breaking,
    because it exploits the errors in the Q-function. Twin Delayed DDPG (TD4) 
    is an algorithm that addresses this issue by introducing three critical tricks:
    Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), 
        and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
    Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) 
        less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.
    Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, 
        to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

    https://arxiv.org/abs/1802.09477

    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        noise_policy (float): the amount of noise to add to each action (before scaling).
        noise_td3 (float): the amount of noise to add to each action in trick three.
        policy_update_td3 (int): Number of timesteps per training update the policy in trick two.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
    """

    def __init__(self,
                 q_1,
                 q_2,
                 policy,
                 discount_factor=0.99,
                 minibatch_size=32,
                 noise_policy=0.1,
                 noise_td3=0.2,
                 policy_update_td3=2,
                 replay_start_size=5000,
                 ):
        # objects
        self.q_1 = q_1
        self.q_2 = q_2
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        action_space = Action.action_space()
        self._noise_policy = Normal(
            0, noise_policy*torch.tensor((
                action_space.high - action_space.low) / 2,
                dtype=torch.float32, device=self.device))

        self._noise_td3 = Normal(
            0, noise_td3*torch.tensor((
                action_space.high - action_space.low) / 2,
                dtype=torch.float32, device=self.device))

        self._policy_update_td3 = policy_update_td3
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward=None):
        if reward is not None:
            samples = Samples(self._states, self._actions, reward, states)
            self.replay_buffer.store(samples)
        self._states = states
        actions = self.policy.no_grad(states.to(self.device))
        actions = actions + self._noise_policy.sample([actions.shape[0]])
        self._actions = Action(actions).to("cpu")
        return self._actions

    def train(self):
        self._train_count += 1
        if self.should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)

            # Trick Three: Target Policy Smoothing
            next_actions = self.policy.target(next_states)
            next_actions += self._noise_td3.sample([next_actions.shape[0]])

            # train q-network
            # Trick One: clipped double q learning
            q_targets = rewards + self.discount_factor * \
                torch.min(self.q_1.target(next_states, Action(next_actions)),
                          self.q_2.target(next_states, Action(next_actions)))
            q_1_values = self.q_1(states, actions)
            self.q_1.reinforce(weighted_mse_loss(
                q_1_values, q_targets, weights))
            q_2_values = self.q_2(states, actions)
            self.q_2.reinforce(weighted_mse_loss(
                q_2_values, q_targets, weights))

            # update priorities
            td_errors = (q_targets - q_1_values).abs()
            self.replay_buffer.update_priorities(indexes, td_errors.cpu())

            # train policy
            # Trick Two: delayed policy updates
            if self._train_count % self._policy_update_td3 == 0:
                greedy_actions = self.policy(states)
                loss = -self.q_1(states, Action(greedy_actions)).mean()
                self.policy.reinforce(loss)

            # additional debugging info
            self.writer.add_scalar('loss/td_error', td_errors.mean())

            self.writer.train_steps += 1

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        q_model = deepcopy(self.q_1.model)
        policy_target_model = deepcopy(self.policy._target._target)
        q_target_model = deepcopy(self.q_1._target._target)
        noise = Normal(0, self._noise_policy.stddev.to("cpu"))
        return DDPGLazyAgent(policy_model=policy_model.to("cpu"),
                             policy_target_model=policy_target_model.to("cpu"),
                             q_model=q_model.to("cpu"),
                             q_target_model=q_target_model.to("cpu"),
                             discount_factor=self.discount_factor,
                             noise=noise,
                             use_apex=use_apex(),
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
