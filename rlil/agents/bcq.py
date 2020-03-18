import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import State, action_decorator, Action
from rlil import nn
from ._agent import Agent


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

    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        vae (AutoEncoder): An approximation of the VAE.
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        replay_buffer (ReplayBuffer): The experience replay buffer with enough data.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        noise_policy (float): the amount of noise to add to each action (before scaling).
        noise_td3 (float): the amount of noise to add to each action in trick three.
        policy_update_td3 (int): Number of timesteps per training update the policy in trick two.
    """

    def __init__(self,
                 q_1,
                 q_2,
                 vae,
                 policy,
                 replay_buffer,
                 discount_factor=0.99,
                 minibatch_size=32,
                 noise_policy=0.1,
                 noise_td3=0.2,
                 policy_update_td3=2,
                 device=torch.device("cpu")
                 ):
        # objects
        self.q_1 = q_1
        self.q_2 = q_2
        self.vae = vae
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.device = device
        # hyperparameters
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        action_space = Action.action_space()
        self._noise_policy = Normal(
            0, noise_policy * torch.tensor((action_space.high - action_space.low) / 2).to(self.device))
        self._noise_td3 = Normal(
            0, noise_td3 * torch.tensor((action_space.high - action_space.low) / 2).to(self.device))
        self._policy_update_td3 = policy_update_td3
        self._low = torch.tensor(action_space.low, device=self.device)
        self._high = torch.tensor(action_space.high, device=self.device)
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward):
        self._states = states
        self._actions = self._choose_actions(states)
        return self._actions

    @action_decorator
    def _choose_actions(self, states: State, actions: Action):
        # choose vae action
        mean, log_var = self.vae.encode(states.to(self.device), actions.to(self.device))
        z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
        vae_action = Action(self.vae.decode(states, z))

        # choose normal action
        actions = self.policy.eval(states.to(self.device), vae_action.to(self.device))
        actions = actions + self._noise_policy.sample([actions.shape[0]])
        return actions.to("cpu")

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size, device=self.device)

            # training vae
            mean, log_var = self.vae.encode(states.to(self.device), actions.to(self.device))
            z = mean + (0.5 * log_var).exp() * torch.randn_like(log_var)
            vae_action = Action(self.vae.decode(states, z))
            vae_loss = mse_loss(actions.features, vae_action) \
                + nn.kl_loss(mean, log_var)
            self.vae.reinforce(vae_loss)

            # training critic
            # Trick Three: Target Policy Smoothing
            next_actions = self.policy.target(next_states)
            next_actions += self._noise_td3.sample([next_actions.shape[0]])
            next_actions = torch.min(next_actions, self._high)
            next_actions = torch.max(next_actions, self._low)

            # train q-network
            # Trick One: clipped double q learning
            q_targets = rewards + self.discount_factor * \
                torch.min(self.q_1.target(next_states, next_actions),
                          self.q_2.target(next_states, next_actions))
            self.q_1.reinforce(
                mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(
                mse_loss(self.q_2(states, actions), q_targets))

            # train policy
            # Trick Two: delayed policy updates
            if self._train_count % self._policy_update_td3 == 0:
                greedy_actions = self.policy(states)
                loss = -self.q_1(states, greedy_actions).mean()
                self.policy.reinforce(loss)
                self.policy.zero_grad()
            self.q_1.zero_grad()
            self.q_2.zero_grad()

    def _should_train(self):
        self._train_count += 1
        return True
