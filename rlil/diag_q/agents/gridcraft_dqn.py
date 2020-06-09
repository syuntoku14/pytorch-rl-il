import torch
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex)
from rlil.policies import GreedyPolicy
from rlil.environments import Action
from rlil.agents import DQN
from rlil.diag_q.gridcraft.utils import get_all_states
from rlil.diag_q.gridcraft.wrappers import ObsWrapper
from copy import deepcopy
import os


class GridCraftDQN(DQN):
    '''
    This class is for debugging q learning using gridcraft.
    '''

    def __init__(self, env: ObsWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute true q values
        self.wrapped_env = env.wrapped_env
        self.wrapped_env.set_qval(gamma=self.discount_factor, K=1000)
        self.true_q_image = self.wrapped_env.plot_qval(
            self.wrapped_env.qval, return_image=True)
        self.all_states = get_all_states(env, append_time=True).to(self.device)
        self.writer.add_text("maze", self.wrapped_env.gs.string)
        self.writer.add_image(
            "qval/true", self.true_q_image, interval_scale=-1)

    def train(self):
        if self.should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            targets = rewards + self.discount_factor * \
                torch.max(self.q.target(next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets, weights)
            # backward pass
            self.q.reinforce(loss)
            # update epsilon greedy
            self.epsilon.update()
            self.policy.set_epsilon(self.epsilon.get())

            # additional debugging info
            if self.writer.train_steps % 100 == 0:
                all_qvals = self.q(self.all_states).detach().cpu().numpy()
                q_image = self.wrapped_env.plot_qval(all_qvals, return_image=True)
                self.writer.add_image(
                    "qval/predict", q_image, interval_scale=-1)

            self.writer.train_steps += 1
