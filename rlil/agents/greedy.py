from rlil.agents import Agent
from rlil.environments import Action, action_decorator
from rlil.memory import ExperienceReplayBuffer
import torch
import gym
import os


class GreedyAgent(Agent):
    def __init__(
            self,
            action_space,
            feature=None,
            q=None,
            policy=None,
            device=torch.device("cpu")
    ):
        self.action_space = action_space
        self.feature = feature
        self.policy = None
        self.device = device
        self.replay_buffer = ExperienceReplayBuffer(size=1e5)
        if policy:
            self.policy = policy
        else:
            self.policy = q
        if not self.policy:
            raise TypeError(
                'GreedyAgent must have either policy or q function')
        self._state = None
        self._action = None

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._state = state
        with torch.no_grad():
            if self.feature:
                state = self.feature(state)
            if isinstance(self.action_space, gym.spaces.Discrete):
                self._action = self.choose_discrete(state)
            elif isinstance(self.action_space, gym.spaces.Box):
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
        for filename in os.listdir(dirname):
            if filename == 'feature.pt':
                feature = torch.load(os.path.join(
                    dirname, filename)).to(self.device)
            if filename == 'policy.pt':
                policy = torch.load(os.path.join(
                    dirname, filename)).to(self.device)
            if filename in ('q.pt', 'q_dist.pt'):
                q = torch.load(os.path.join(dirname, filename)).to(self.device)

        agent = GreedyAgent(
            env.action_space,
            feature=feature,
            policy=policy,
            q=q,
        )

        return agent
