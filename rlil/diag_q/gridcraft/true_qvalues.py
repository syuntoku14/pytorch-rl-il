import os
import sys
import numpy as np
import hashlib
from .grid_env import GridEnv
from .utils import one_hot_to_flat, flat_to_one_hot


def get_hashname(obj):
    return hex(hash(obj) % ((sys.maxsize+1)*2))[2:]


def hash_env(gridspec, env_args, gamma):
    #d = (get_hashname(gridspec), tuple(env_args.items()), gamma)
    m = hashlib.md5()
    m.update(get_hashname(gridspec).encode('utf-8'))
    m.update(str(tuple(env_args.items())).encode('utf-8'))
    m.update(str(gamma).encode('utf-8'))
    return m.hexdigest()+'.csv'


def dense_tabular_solver(gridspec, env_args, K=50, gamma=0.95, verbose=False):
    env = GridEnv(gridspec, **env_args)
    # solve with Q-iteration
    dS = len(env.gs)
    dA = env.action_space.n

    transition_matrix = env.transition_matrix
    rew_matrix = env.rew_matrix
    r_sa = np.sum(transition_matrix * rew_matrix, axis=2)

    q_values = np.zeros((dS, dA))
    prev_diff = 1.0
    for k in range(K):
        v_fn = np.max(q_values, axis=1)  # dO
        new_q = r_sa + gamma * transition_matrix.dot(v_fn)
        diff = np.max(np.abs(new_q - q_values))
        if verbose:
            print(k, 'InfNorm:', diff, 'ContractionFactor:',
                  '%0.4f' % (diff/prev_diff))
        q_values = new_q
        prev_diff = diff
    return q_values
