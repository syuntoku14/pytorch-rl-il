import torch
import numpy as np
import os
from scipy.special import logsumexp as sp_lse
from rlil.diag_q.envs import (tabular_env, time_limit_wrapper)
from rlil.diag_q.compute_q import q_iteration
from rlil.environments import GymEnvironment, ENVS, State, Action
from rlil.presets import continuous


STATE_DISC = 32
ACTION_DISC = 5
MAX_STEPS = 200

ORIGIN_ENV = tabular_env.InvertedPendulum(
    state_discretization=STATE_DISC, action_discretization=ACTION_DISC)
random_init = {
    i: 1.0 / ORIGIN_ENV.num_states for i in range(ORIGIN_ENV.num_states)}
ORIGIN_ENV = tabular_env.InvertedPendulum(
    state_discretization=STATE_DISC, action_discretization=ACTION_DISC, init_dist=random_init)
ENV = time_limit_wrapper.TimeLimitWrapper(ORIGIN_ENV, MAX_STEPS)
RLIL_ENV = GymEnvironment(ENVS["pendulum"], append_time=True)


def compute_true_action_values(qval_path=None):
    """
    compute true action values using q_iteration
    Args:
        qval_path (string): 
        If qval_path is not None, this function return saved q_vals.

    Returns:
        qvals (np.array)
    """

    if qval_path is not None:
        qvals = np.load(qval_path)
        return qvals

    params = {
        'num_itrs': 300,
        'ent_wt': 0.0,
        'discount': 0.99,
    }

    qvals = q_iteration.softq_iteration(ENV, **params, verbose=True)
    np.save("pendulum_q.npy", qvals)
    return qvals


def compute_true_state_values(qvals):
    """
    Return state values by taking max over qvals
    """
    vvals = np.zeros((MAX_STEPS, STATE_DISC, STATE_DISC))
    for wrapped_s in range(ENV.num_states - 1):
        time, s = ENV.unwrap_state(wrapped_s)
        th, thv = ENV.wrapped_env.th_thv_from_id(s)
        th, thv = ENV.wrapped_env.disc_th_thv(th, thv)
        v = qvals[wrapped_s].max()
        vvals[time, th, thv] = v
    return vvals


def get_all_states():
    """
    Return State for computing q or v.
    """
    s1, s2, s3, s4 = [], [], [], []
    for wrapped_s in range(ENV.num_states - 1):
        time, s = ENV.unwrap_state(wrapped_s)
        th, thv = ENV.wrapped_env.th_thv_from_id(s)
        s1.append(np.cos(th))
        s2.append(np.sin(th))
        s3.append(thv)
        s4.append(min(time / MAX_STEPS, 1.0))

    states = torch.cat((torch.tensor(s1, dtype=torch.float32).unsqueeze(1),
                        torch.tensor(s2, dtype=torch.float32).unsqueeze(1),
                        torch.tensor(s3, dtype=torch.float32).unsqueeze(1),
                        torch.tensor(s4, dtype=torch.float32).unsqueeze(1)), dim=1)
    return State(states)


def predict_action_values(q_func):
    """
    Predict action values of all states and actions.
    Returns:
        pred_qvals (np.array): Predicted action values
    """
    states = get_all_states()
    torques = torch.tensor([ENV.wrapped_env.torque_from_id(
        i) for i in range(ACTION_DISC)], dtype=torch.float32).unsqueeze(0)
    torques = torch.repeat_interleave(torques, states.shape[0], dim=0)
    actions_list = [Action(torques[:, i].unsqueeze(1))
                    for i in range(ACTION_DISC)]
    pred_q_list = []
    for i in range(ACTION_DISC):
        pred_q_list.append(
            q_func(states.to(q_func.device), actions_list[0].to(q_func.device)))
    pred_qvals = torch.stack(pred_q_list, dim=1)
    return pred_qvals


def predict_state_values(q_func):
    """
    Predict state values of all states and actions.
    Returns:
        pred_vvals (np.array): Predicted state values
    """
    pred_qvals = predict_action_values(q_func)
    values = torch.max(pred_qvals, dim=1)[0]

    pred_vvals = np.zeros((MAX_STEPS, STATE_DISC, STATE_DISC))
    for wrapped_s in range(ENV.num_states - 1):
        time, s = ENV.unwrap_state(wrapped_s)
        th, thv = ENV.wrapped_env.th_thv_from_id(s)
        th, thv = ENV.wrapped_env.disc_th_thv(th, thv)
        pred_vvals[time, th, thv] = values[wrapped_s]
    return pred_vvals


def predict_state_values_ppo(feature_nw, v_func):
    """
    This function is for ppo to predict state values.
    """
    states = get_all_states()
    features = feature_nw(states.to(feature_nw.device))
    values = v_func(features)

    pred_vvals = np.zeros((MAX_STEPS, STATE_DISC, STATE_DISC))
    for wrapped_s in range(ENV.num_states - 1):
        time, s = ENV.unwrap_state(wrapped_s)
        th, thv = ENV.wrapped_env.th_thv_from_id(s)
        th, thv = ENV.wrapped_env.disc_th_thv(th, thv)
        pred_vvals[time, th, thv] = values[wrapped_s]
    return pred_vvals


def state_to_th_thv_time(states):
    theta = torch.atan2(
        states.raw[:, 1], states.raw[:, 0]).cpu().detach().numpy()
    theta_v = states.raw[:, 2].cpu().detach().numpy()
    time_step = (states.raw[:, -1] *
                 MAX_STEPS).floor().int().cpu().detach().numpy()
    return theta, theta_v, time_step


def compute_true_action_values_from_samples(full_qvals, states, actions):
    theta, theta_v, time_step = state_to_th_thv_time(states)
    torques = actions.features.cpu().detach().numpy()
    qvals = np.zeros(states.shape[0])
    for i, (th, thv, t, trq) in enumerate(zip(theta, theta_v, time_step, torques)):
        s_idx = ENV.wrapped_env.id_from_th_thv(th, thv)
        s_idx = ENV.wrap_state(s_idx, t)
        a_idx = ENV.wrapped_env.id_from_torque(trq)
        qvals[i] = full_qvals[s_idx][a_idx]
    return qvals


def logsumexp(q, alpha=1.0, axis=1):
    if alpha == 0:
        return np.max(q, axis=axis)
    return alpha*sp_lse((1.0/alpha)*q, axis=axis)


def get_policy(q_fn, ent_wt=1.0):
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    if ent_wt == 0:
        pol_probs = adv_rew
        pol_probs[pol_probs >= 0] = 1.0
        pol_probs[pol_probs < 0] = 0.0
    else:
        pol_probs = np.exp((1.0/ent_wt)*adv_rew)
    pol_probs /= np.sum(pol_probs, axis=1, keepdims=True)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs


def sample_visitation(q_fn, sample_iters=1000):
    policy = get_policy(q_fn)
    visitation = np.zeros((MAX_STEPS, STATE_DISC, STATE_DISC))
    for _ in range(sample_iters):
        done = False
        ENV.reset()
        wrapped_s = ENV.get_state()
        while not done:
            # record visitation
            time, s = ENV.unwrap_state(wrapped_s)
            th, thv = ENV.wrapped_env.th_thv_from_id(s)
            th, thv = ENV.wrapped_env.disc_th_thv(th, thv)
            visitation[time, th, thv] += 1

            # do step
            a = np.random.choice(5, p=policy[wrapped_s])
            _, _, done, _ = ENV.step(a)
            wrapped_s = ENV.get_state()
    return visitation / sample_iters
