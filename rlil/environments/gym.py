from .action import Action
from .state import State
from .base import Environment
import torch
import numpy as np
import gym
from gym.wrappers import TimeLimit
from rlil.initializer import get_device, is_debug_mode
gym.logger.set_level(40)


class GymEnvironment(Environment):
    """
    When append_time is True, timestep is appended to the output State.
    """

    def __init__(self,
                 env,
                 device=torch.device("cpu"),
                 append_time=False):

        self.device = device
        self._name = env
        if isinstance(env, str):
            env = gym.make(env)
        self._env = env
        self._state: State = None
        self._action: Action = None
        self._reward = None
        self._done = True
        self._info = None
        self._append_time = None
        self._observation_space = self._env.observation_space
        self.set_append_time(append_time)
        # lazy init
        self._init = False
        self._done_mask = None
        self._not_done_mask = None

        # set action_space
        Action.set_action_space(env.action_space)

    def set_append_time(self, append_time):
        self._append_time = append_time
        if append_time:
            assert isinstance(self._env, TimeLimit), \
                "env must be TimeLimit when append_time is True."
            assert len(self._env.observation_space.shape) == 1, \
                "observation_space must be one dimension when append_time is True."
            obs_space = type(self._env.observation_space)
            low = self._env.observation_space.low
            high = self._env.observation_space.high
            self._observation_space = \
                obs_space(low=np.hstack((low, np.array([0, ]))),
                          high=np.hstack((low, np.array([1.0, ]))))

    @property
    def name(self):
        return self._name

    def reset(self):
        self._lazy_init()
        state = self._env.reset()
        self._state = self._make_state(state, 0)
        self._action = None
        self._reward = torch.tensor([0], dtype=torch.float32,
                                    device=self.device)
        self._done = False
        return self._state

    def step(self, action):
        if is_debug_mode():
            assert isinstance(
                action, Action), "Input invalid action type {}. action must be Action".format(type(action))
        state, reward, done, info = self._env.step(
            self._convert_action(action))
        self._state = self._make_state(state, done, info)
        self._action = action
        self._reward = self._convert_reward(reward)
        self._done = done
        return self._state, self._reward

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self):
        return GymEnvironment(self._name,
                              device=self.device,
                              append_time=self._append_time)

    @property
    def state_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    @property
    def info(self):
        return self._state.info

    @property
    def env(self):
        return self._env

    def _lazy_init(self):
        if not self._init:
            # predefining these saves performance on tensor creation
            # it actually makes a noticable difference :p
            self._done_mask = torch.tensor(
                [0],
                dtype=torch.bool,
                device=self.device
            )
            self._not_done_mask = torch.tensor(
                [1],
                dtype=torch.bool,
                device=self.device
            )
            self._init = True

    def _make_state(self, raw, done, info=None):
        '''Convert numpy array into State'''
        if self._append_time:
            timestep = float(self._env._elapsed_steps)
            max_step = float(self._env._max_episode_steps)
            raw = np.hstack((raw, np.array([timestep / max_step])))

        return State(
            torch.as_tensor(raw.astype(self.state_space.dtype),
                            device=self.device).unsqueeze(0),
            self._done_mask if done else self._not_done_mask,
            [info]
        )

    def _convert_action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.features.item()
        if isinstance(self.action_space, gym.spaces.Box):
            return action.features.view(self.action_space.shape).cpu().detach().numpy()
        raise TypeError("Unknown action space type")

    def _convert_reward(self, reward):
        if isinstance(reward, torch.Tensor):
            return reward
        else:
            return torch.tensor([reward], dtype=torch.float32,
                                device=self.device)
