from rlil.agents import Agent
from rlil.environments import Action, action_decorator
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import get_device
import torch
import gym
import os


class GreedyAgent(Agent):
    def __init__(
            self,
            feature=None,
            q=None,
            policy=None,
    ):
        self.feature = feature
        self.policy = None
        self.device = get_device()
        self.replay_buffer = ExperienceReplayBuffer(size=5e4)
        if policy:
            self.policy = policy
        else:
            self.policy = q
        if not self.policy:
            raise TypeError(
                'GreedyAgent must have either policy or q function')
        self._state = None
        self._action = None

    def act(self, state, reward=None):
        if reward is not None:
            self.replay_buffer.store(self._state, self._action, reward, state)
        self._state = state
        with torch.no_grad():
            if self.feature:
                state = self.feature(state)
            if isinstance(Action.action_space(), gym.spaces.Discrete):
                self._action = self.choose_discrete(state)
            elif isinstance(Action.action_space(), gym.spaces.Box):
                self._action = self.choose_continuous(state)
            else:
                raise TypeError('Unknown action space')
        return self._action

    @action_decorator
    def choose_discrete(self, state):
        ret = self.policy(state)
        if isinstance(ret, torch.Tensor):
            if len(ret.shape) == 3:  # categorical dqn
                return torch.argmax((ret * self.policy.atoms).sum(dim=2), dim=1).unsqueeze(1)
            return torch.argmax(self.policy(state), dim=1).unsqueeze(1)
        if isinstance(ret, torch.distributions.distribution.Distribution):
            return ret.sample().unsqueeze(1)
        return ret  # unknown type, return it and pray!

    @action_decorator
    def choose_continuous(self, state):
        ret = self.policy(state)
        if isinstance(ret, torch.Tensor):
            return ret
        if isinstance(ret, tuple):
            return ret[0]
        if isinstance(ret, torch.distributions.distribution.Distribution):
            return ret.sample()
        return ret  # unknown type, return it and pray!

    @staticmethod
    def load(dirname, env):
        feature = None
        policy = None
        q = None
        device = get_device()
        for filename in os.listdir(dirname):
            if filename == 'feature.pt':
                feature = torch.load(os.path.join(
                    dirname, filename), map_location=device)
            if filename == 'policy.pt':
                policy = torch.load(os.path.join(
                    dirname, filename), map_location=device)
            if filename in ('q.pt', 'q_dist.pt'):
                q = torch.load(os.path.join(dirname, filename),
                               map_location=device)

        agent = GreedyAgent(
            feature=feature,
            policy=policy,
            q=q,
        )

        return agent

    @staticmethod
    def load_BC(dirname, agent_fn, env):
        policy = agent_fn(env).policy
        device = get_device()

        for filename in os.listdir(dirname):
            if filename == 'BC_state_dict.pt':
                state_dict = torch.load(os.path.join(
                    dirname, filename), map_location=device)
                policy.model.model.load_state_dict(state_dict)

        agent = GreedyAgent(
            policy=policy,
        )

        return agent
