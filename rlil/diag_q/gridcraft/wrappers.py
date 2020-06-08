import numpy as np
from .grid_env import REWARD, GridEnv
from .utils import flat_to_one_hot, one_hot_to_flat
from gym.spaces import Box
from gym import Env, logger
from gym import error


class Wrapper(Env):
    def __init__(self, env=None):
        self._wrapped_env = env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @property
    def base_env(self):
        if isinstance(self.wrapped_env, Wrapper):
            return self.wrapped_env.base_env
        else:
            return self.wrapped_env

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def step(self, action):
        return self.wrapped_env.step(action)

    def reset(self):
        return self.wrapped_env.reset()

    def render(self, **kwargs):
        return self.wrapped_env.render(**kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.wrapped_env)

    def __repr__(self):
        return str(self)


class ObsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def wrap_obs(self, obs, info=None):
        raise NotImplementedError()

    def step(self, action):
        obs, r, done, infos = self.wrapped_env.step(action)
        return self.wrap_obs(obs, info=infos), r, done, infos

    def reset(self, env_info=None):
        if env_info is None:
            env_info = {}
        obs = self.wrapped_env.reset()
        return self.wrap_obs(obs, info=env_info)


class GridObsWrapper(ObsWrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self, *args, **kwargs):
        self.wrapped_env.render()


class CoordinateWiseWrapper(GridObsWrapper):
    def __init__(self, env):
        assert isinstance(env, GridEnv)
        super().__init__(env)
        self.gs = env.gs
        self.dO = self.gs.width+self.gs.height

        self.observation_space = Box(0, 1, (self.dO, ))

    def wrap_obs(self, obs, info=None):
        xy = self.gs.idx_to_xy(obs)
        x = flat_to_one_hot(xy[0], self.gs.width)
        y = flat_to_one_hot(xy[1], self.gs.height)
        obs = np.hstack([x, y])
        return obs

    def unwrap_obs(self, obs, info=None):

        if len(obs.shape) == 1:
            x = obs[:self.gs.width]
            y = obs[self.gs.width:]
            x = one_hot_to_flat(x)
            y = one_hot_to_flat(y)
            state = self.gs.xy_to_idx(np.c_[x, y])
            return flat_to_one_hot(state, self.dO)
        else:
            raise NotImplementedError()


class RandomObsWrapper(GridObsWrapper):
    def __init__(self, env, obs_dim):
        assert isinstance(env, GridEnv)
        super().__init__(env)
        self.gs = env.gs
        self.obs_dim = obs_dim
        self.obs_matrix = np.random.randn(len(self.gs), self.obs_dim)
        self.observation_space = Box(np.min(self.obs_matrix), np.max(self.obs_matrix),
                                     shape=(self.obs_dim, ), dtype=np.float32)

    def wrap_obs(self, obs, info=None):
        return self.obs_matrix[obs]

    def unwrap_obs(self, obs, info=None):
        raise NotImplementedError()
