import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import State, action_decorator, Action
from rlil.initializer import get_device, get_replay_buffer
from rlil import nn
from .base import Agent, LazyAgent


class BC(Agent):
    """
    Behavioral Cloning (BC)

    In behavioral cloning, the agent trains a classifier or regressor to
    replicate the expert's policy using the training data 
    both the encountered states and actions.

    Args:
        policy (DeterministicPolicy): 
            An Approximation of a deterministic policy.
        minibatch_size (int): 
            The number of experiences to sample in each training update.
    """

    def __init__(self,
                 policy,
                 minibatch_size=32,
                 ):
        # objects
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        # hyperparameters
        self.minibatch_size = minibatch_size
        action_space = Action.action_space()
        self._low = torch.tensor(action_space.low, device=self.device)
        self._high = torch.tensor(action_space.high, device=self.device)
        self._train_count = 0

    def act(self, states, reward):
        self._states = states
        self._actions = Action(self.policy.eval(states.to(self.device)))
        return self._actions

    def train(self):
        if self._should_train():
            (states, actions, _, _, _) = self.replay_buffer.sample(
                self.minibatch_size)
            policy_actions = Action(self.policy(states))
            loss = mse_loss(policy_actions.features, actions.features)
            self.policy.reinforce(loss)
            self.policy.zero_grad()

    def _should_train(self):
        self._train_count += 1
        return True

    def make_lazy_agent(self, evaluation=False):
        model = deepcopy(self.policy.model)
        return BCLazyAgent(model.to("cpu"), evaluation)


class BCLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """
    def __init__(self, policy_model, evaluation):
        self._replay_buffer = ExperienceReplayBuffer(1e9)
        self._policy_model = policy_model
        self._states = None
        self._actions = None
        self._evaluation = evaluation

    def act(self, states, reward):
        if not self._evaluation:
            self._replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        with torch.no_grad():
            self._actions = Action(self._policy_model(states))
        return self._actions
