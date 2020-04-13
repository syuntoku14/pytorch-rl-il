import torch
from copy import deepcopy
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import Action
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil.memory import ExperienceReplayBuffer
from .base import Agent, LazyAgent


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
        update_frequency (int): Number of timesteps per training update.
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
                 update_frequency=1,
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
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._noise_policy = Normal(
            0, noise_policy*torch.tensor((
                Action.action_space().high
                - Action.action_space().low) / 2).to(self.device))

        self._noise_td3 = Normal(
            0, noise_td3*torch.tensor((
                Action.action_space().high
                - Action.action_space().low) / 2).to(self.device))

        self._policy_update_td3 = policy_update_td3
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        actions = self.policy.eval(states.to(self.device))
        actions = actions + self._noise_policy.sample([actions.shape[0]])
        self._actions = Action(actions).to("cpu")
        return self._actions

    def train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)
            self.writer.train_frames += len(states)

            # Trick Three: Target Policy Smoothing
            next_actions = self.policy.target(next_states)
            next_actions += self._noise_td3.sample([next_actions.shape[0]])

            # train q-network
            # Trick One: clipped double q learning
            q_targets = rewards + self.discount_factor * \
                torch.min(self.q_1.target(next_states, Action(next_actions)),
                          self.q_2.target(next_states, Action(next_actions)))
            self.q_1.reinforce(
                mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(
                mse_loss(self.q_2(states, actions), q_targets))

            # train policy
            # Trick Two: delayed policy updates
            if self._train_count % self._policy_update_td3 == 0:
                greedy_actions = self.policy(states)
                loss = -self.q_1(states, Action(greedy_actions)).mean()
                self.policy.reinforce(loss)
                self.policy.zero_grad()
            self.q_1.zero_grad()
            self.q_2.zero_grad()

    def _should_train(self):
        self._train_count += 1
        return len(self.replay_buffer) > self.replay_start_size and self._train_count % self.update_frequency == 0

    def make_lazy_agent(self, evaluation=False):
        model = deepcopy(self.policy.model)
        noise = Normal(0, self._noise_policy.stddev.to("cpu"))
        return TD3LazyAgent(model.to("cpu"), noise, evaluation)


class TD3LazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, noise_policy, evaluation):
        self._replay_buffer = ExperienceReplayBuffer(1e9)
        self._policy_model = policy_model
        self._noise_policy = noise_policy
        self._states = None
        self._actions = None
        self._evaluation = evaluation

    def act(self, states, reward):
        if not self._evaluation:
            self._replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        with torch.no_grad():
            actions = self._policy_model(states)
            if not self._evaluation:
                actions += self._noise_policy.sample([actions.shape[0]])
        self._actions = Action(actions)
        return self._actions
