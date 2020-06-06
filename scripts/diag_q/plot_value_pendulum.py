import gym
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import torch
import argparse
import numpy as np
import os
from rlil.environments import GymEnvironment, ENVS, State, Action
from rlil.presets import continuous
from rlil.initializer import set_device
from rlil.utils.diag_q.pendulum_util import *
matplotlib.use('Agg')


def plot_state_values(fig, vvals, row_id, num_rows, num_cols=5, start_step=0,
                      vmin=None, vmax=None):
    vmin = vvals.min() if vmin is None else vmin
    vmax = vvals.max() if vmax is None else vmax
    step_interval = (MAX_STEPS - start_step) / num_cols

    th_ticks, thv_ticks = [], []
    for i in range(STATE_DISC):
        th, thv = ENV.wrapped_env.from_disced_th_thv(i, i)
        th_ticks.append(round(th, 3))
        thv_ticks.append(round(thv, 3))

    # plot step == start_step
    data = pd.DataFrame(
        vvals[start_step, :, :], index=th_ticks,
        columns=thv_ticks).rolling(3).median().rolling(3, axis=1).median(axis=1)
    ax = fig.add_subplot(num_rows, num_cols, row_id * num_cols + 1)
    sns.heatmap(data, ax=ax, cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title("time step: {}".format(start_step))
    ax.set_ylabel(r"$\theta$")
    ax.set_xlabel(r"$\dot{\theta}$")

    # plot step == start_step + step_interval ~ MAX_STEPS - 1
    for i in range(2, num_cols):
        idx = int(i * step_interval) + start_step
        data = pd.DataFrame(
            vvals[idx, :, :], index=th_ticks,
            columns=thv_ticks).rolling(3).median().rolling(3, axis=1).median(axis=1)
        ax = fig.add_subplot(num_rows, num_cols, row_id * num_cols + i)
        sns.heatmap(data, ax=ax, cbar=False,
                    yticklabels=False, vmin=vmin, vmax=vmax)
        ax.set_title("time step: {}".format(idx))
        ax.set_xlabel(r"$\dot{\theta}$")

    # plot step == MAX_STEPS - 1
    data = pd.DataFrame(
        vvals[-1, :, :], index=th_ticks,
        columns=thv_ticks).rolling(3).median().rolling(3, axis=1).median(axis=1)
    ax = fig.add_subplot(num_rows, num_cols, (row_id + 1) * num_cols)
    sns.heatmap(data, ax=ax,
                yticklabels=False, vmin=vmin, vmax=vmax)
    ax.set_title("time step: {}".format(MAX_STEPS-1))
    ax.set_xlabel(r"$\dot{\theta}$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qval_path", type=str, default="./pendulum_q.npy")
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_cols", type=int, default=5)
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share_cbar", action="store_true")
    args = parser.parse_args()

    set_device(args.device)

    # setup figure
    agent_dirs = []
    figure_title = "share_cbar_" if args.share_cbar else ""
    if args.runs_dir is not None:
        runs_dir = Path(args.runs_dir)
        figure_title += runs_dir.name + "_v_{}-200.png".format(args.start_step)
        agent_dirs = sorted([agent_dir for agent_dir in runs_dir.iterdir()
                             if agent_dir.is_dir()])
    figure_title += "v_{}-200.png".format(args.start_step)

    num_rows = 1 + len(agent_dirs)
    fig, big_axes = plt.subplots(nrows=num_rows, ncols=1,
                                 figsize=(args.num_cols*6, num_rows*6),
                                 sharey=True)
    if num_rows == 1:
        big_axes = [big_axes]

    for i in range(num_rows):
        big_axes[i].tick_params(labelcolor=(1., 1., 1., 0.0),
                                top='off', bottom='off', left='off', right='off')
        big_axes[i]._frameon = False

    # plot true v values
    print("Plotting true state values ...")
    true_q_vals = compute_true_action_values(args.qval_path)
    true_v_vals = compute_true_state_values(true_q_vals)

    big_axes[0].set_title("True state value map\n\n", fontsize="25")
    plot_state_values(fig, true_v_vals,
                      row_id=0,
                      num_rows=num_rows,
                      num_cols=args.num_cols,
                      start_step=args.start_step)

    # plot agent v values
    for i, agent_dir in enumerate(agent_dirs):
        agent_name = agent_dir.name

        if args.agent_name is not None:
            agent_name = args.agent_name + "_" + agent_name

        print("Plotting {}...".format(agent_name))
        agent_fn = getattr(continuous, agent_name.split("_")[0])()
        agent = agent_fn(RLIL_ENV)
        agent.load(agent_dir)
        if "ppo" in agent_name:
            feature_nw = agent.feature_nw
            v_func = agent.v
            pred_v_vals = predict_state_values_ppo(feature_nw, v_func)
        else:
            try:
                q_func = agent.q
            except AttributeError:
                q_func = agent.q_1
            pred_v_vals = predict_state_values(q_func)
        big_axes[i + 1].set_title(
            agent_name + " state value map\n\n", fontsize="25")

        vmin = true_v_vals.min() if args.share_cbar else None
        vmax = true_v_vals.max() if args.share_cbar else None
        plot_state_values(fig, pred_v_vals,
                          row_id=i + 1,
                          num_rows=num_rows,
                          num_cols=args.num_cols,
                          start_step=args.start_step,
                          vmin=vmin, vmax=vmax)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_title))
