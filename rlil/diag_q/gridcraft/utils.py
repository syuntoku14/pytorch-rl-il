import numpy as np
from rlil.environments import State


def flat_to_one_hot(val, ndim):
    """

    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    >>> flat_to_one_hot(np.array([2, 4, 3]), ndim=5)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.]])
    """
    shape = np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v


def one_hot_to_flat(val):
    """
    >>> one_hot_to_flat(np.array([0,0,0,0,1]))
    4
    >>> one_hot_to_flat(np.array([0,0,1,0]))
    2
    >>> one_hot_to_flat(np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0]]))
    array([2, 0, 1])
    """
    idxs = np.array(np.where(val == 1.0))[-1]
    if len(val.shape) == 1:
        return int(idxs)
    return idxs


def compute_policy_deterministic(q_values, eps_greedy=0.0):
    policy_probs = np.zeros_like(q_values)
    policy_probs[np.arange(policy_probs.shape[0]), np.argmax(q_values, axis=1)] = 1.0 - eps_greedy
    policy_probs += eps_greedy / (policy_probs.shape[1])
    return policy_probs


def compute_visitation(env, policy, discount=1.0, T=50):
    dS = env.num_states
    dA = env.num_actions
    state_visitation = np.zeros((dS, 1))
    for (state, prob) in env.initial_state_distribution().items():
        state_visitation[state] = prob
    t_matrix = env.transition_matrix  # S x A x S
    sa_visit_t = np.zeros((dS, dA, T))

    norm_factor = 0.0
    for i in range(T):
        sa_visit = state_visitation * policy
        cur_discount = (discount ** i)
        sa_visit_t[:, :, i] = cur_discount * sa_visit
        norm_factor += cur_discount
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
    return np.sum(sa_visit_t, axis=2) / norm_factor


def get_all_states(env, append_time=False):
    # env: ObsWrapper
    states = []
    for s in range(env.wrapped_env.num_states):
        s = env.wrap_obs(s)
        if append_time:
            s = np.hstack((s, [0]))
        states.append(np.expand_dims(s, axis=0))
    states = np.vstack(states)
    return State.from_numpy(states)