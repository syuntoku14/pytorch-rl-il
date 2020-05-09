import torch
import gym
import torch
import numpy as np


class PendulumReward:
    def __call__(self, states, next_states, actions):
        # reward function of Pendulum-v0
        thetas = torch.atan2(states.features[:, 1], states.features[:, 0])
        theta_dots = states.features[:, 2]

        def angle_normalize(x):
            return (((x+np.pi) % (2*np.pi)) - np.pi)

        costs = angle_normalize(thetas) ** 2 \
            + .1 * theta_dots ** 2 \
            + .001*(actions.features.squeeze()**2)
        return -costs
