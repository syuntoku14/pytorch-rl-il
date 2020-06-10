import os
import sys
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from .grid_env import GridEnv
from .utils import one_hot_to_flat, flat_to_one_hot


def get_hashname(obj):
    return hex(hash(obj) % ((sys.maxsize+1)*2))[2:]


def hash_env(gridspec, env_args, gamma):
    # d = (get_hashname(gridspec), tuple(env_args.items()), gamma)
    m = hashlib.md5()
    m.update(get_hashname(gridspec).encode('utf-8'))
    m.update(str(tuple(env_args.items())).encode('utf-8'))
    m.update(str(gamma).encode('utf-8'))
    return m.hexdigest()+'.csv'


class Solver(ABC):
    def __init__(self, gridenv):
        self.env = gridenv
        self.gs = gridenv.gs
        self.dS = len(self.gs)
        self.dA = self.env.action_space.n
        self.transition_matrix = self.env.transition_matrix  # SxAxS
        self.rew_matrix = self.env.rew_matrix  # SxAxS
        self.values = None  # SxA
        self.policy = None  # SxA
        self.state_visitation = None  # S

    @abstractmethod
    def solve(self):
        # compute values and set it to self.values
        pass

    @abstractmethod
    def compute_policy(self):
        pass

    def plot_values(self, values, title=None, return_image=False):
        from .plotter import plot_grid_values
        return plot_grid_values(self.gs, values, title, return_image)

    def compute_visitation(self, policy, discount=1.0, T=50):
        state_visitation = np.zeros((self.dS, 1))
        for (state, prob) in self.env.initial_state_distribution().items():
            state_visitation[state] = prob
        t_matrix = self.transition_matrix  # S x A x S
        sa_visit_t = np.zeros((self.dS, self.dA, T))

        norm_factor = 0.0
        for i in range(T):
            sa_visit = state_visitation * policy
            cur_discount = (discount ** i)
            sa_visit_t[:, :, i] = cur_discount * sa_visit
            norm_factor += cur_discount
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        state_visitation = np.sum(sa_visit_t, axis=2) / norm_factor
        self.state_visitation = state_visitation
        return state_visitation


class ValueIterationSolver(Solver):
    def solve(self, K=50, gamma=0.95, verbose=False):
        # solve with value iteration
        q_values = np.zeros((self.dS, self.dA))  # SxA
        prev_diff = 1.0
        for k in range(K):
            v_fn = np.max(q_values, axis=1)  # S
            new_q = np.sum(self.transition_matrix *
                           (self.rew_matrix + gamma*v_fn), axis=2)  # SxA
            diff = np.max(np.abs(new_q - q_values))
            if verbose:
                print(k, 'InfNorm:', diff, 'ContractionFactor:',
                      '%0.4f' % (diff/prev_diff))
            q_values = new_q
            prev_diff = diff

        self.values = q_values
        return q_values

    def compute_policy(self, q_values, eps_greedy=0.0):
        # return epsilon-greedy policy
        policy_probs = np.zeros_like(q_values)
        policy_probs[np.arange(policy_probs.shape[0]), np.argmax(
            q_values, axis=1)] = 1.0 - eps_greedy
        policy_probs += eps_greedy / (policy_probs.shape[1])
        self.policy = policy_probs
        return policy_probs

