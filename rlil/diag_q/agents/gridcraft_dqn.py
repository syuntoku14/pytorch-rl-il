import torch
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex)
from rlil.policies import GreedyPolicy
from rlil.environments import Action, State
from rlil.agents import DQN
from rlil.diag_q.gridcraft.wrappers import ObsWrapper
from rlil.diag_q.gridcraft.solver import ValueIterationSolver
from copy import deepcopy
import os


class GridCraftDQN(DQN):
    '''
    This class is for debugging q learning using gridcraft.
    '''

    def __init__(self, env: ObsWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute true q values
        self.solver = ValueIterationSolver(env.wrapped_env)
        self.solver.solve(gamma=self.discount_factor, K=1000)
        # q image
        self.true_q_image = self.solver.plot_values(
            self.solver.values, title="Ground truth", return_image=True)
        # policy image
        true_policy = self.solver.compute_policy(self.solver.values)
        self.true_policy_image = self.solver.plot_values(
            true_policy, title="Ground truth", return_image=True)
        # visitation image
        true_visitation = self.solver.compute_visitation(true_policy)
        self.true_visitation_image = self.solver.plot_values(
            true_visitation, title="Ground truth", return_image=True)

        self.all_states = State.from_numpy(
            env.get_all_states(append_time=True)).to(self.device)
        self.writer.add_text("maze", self.solver.gs.string)

    def train(self):
        if self.should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            targets = rewards + (self.discount_factor**self.n_step) * \
                torch.max(self.q.target(next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets, weights)
            # backward pass
            self.q.reinforce(loss)
            # update epsilon greedy
            self.epsilon.update()
            self.policy.set_epsilon(self.epsilon.get())

            # additional debugging info
            if self.writer.train_steps % 500 == 0:
                # plot q values
                all_qvals = self.q(self.all_states).detach().cpu().numpy()
                q_image = self.solver.plot_values(
                    all_qvals, title="DQN", return_image=True)
                q_image = torch.cat((self.true_q_image, q_image), dim=2)
                self.writer.add_image(
                    "Q", q_image, interval_scale=-1)

                # plot policy
                dqn_policy = self.solver.compute_policy(all_qvals)
                dqn_policy_image = self.solver.plot_values(
                    dqn_policy, title="DQN", return_image=True)
                policy_image = torch.cat(
                    (self.true_policy_image, dqn_policy_image), dim=2)
                self.writer.add_image(
                    "policy", policy_image, interval_scale=-1)

                # plot visitation
                dqn_visitation = self.solver.compute_visitation(dqn_policy)
                dqn_visitation_image = self.solver.plot_values(
                    dqn_visitation, title="DQN", return_image=True)
                visitation_image = torch.cat(
                    (self.true_visitation_image, dqn_visitation_image), dim=2)
                self.writer.add_image(
                    "visitation", visitation_image, interval_scale=-1)

            self.writer.train_steps += 1
