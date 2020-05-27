import torch
import os
from copy import deepcopy
from torch.distributions.normal import Normal
from rlil.environments import Action
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil.memory import ExperienceReplayBuffer
from rlil import nn
from .td3 import TD3, LazyAgent


class NoisyTD3(TD3):
    """
    Twin Dueling DDPG (TD3) with noisy network.
    TD3: https://arxiv.org/abs/1802.09477
    Noisy Network: https://arxiv.org/abs/1706.10295

    Args:
        q_1 (QContinuous): An Approximation of the continuous action Q-function.
        q_2 (QContinuous): An Approximation of the continuous action Q-function.
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
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
        self._noise_td3 = Normal(
            0, noise_td3*torch.tensor(
                (Action.action_space().high - Action.action_space().low) / 2,
                dtype=torch.float32, device=self.device))

        self._policy_update_td3 = policy_update_td3
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        actions = self.policy.no_grad(states.to(self.device))
        self._actions = Action(actions).to("cpu")
        return self._actions

    def make_lazy_agent(self,
                        evaluation=False,
                        store_samples=True):
        model = deepcopy(self.policy.model)
        model.apply(nn.perturb_noisy_layers)
        return NoisyTD3LazyAgent(model.to("cpu"),
                                 evaluation=evaluation,
                                 store_samples=store_samples)


class NoisyTD3LazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self,
                 policy_model,
                 *args,
                 **kwargs):
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            actions = self._policy_model(states)
        self._actions = Action(actions)
        return self._actions
