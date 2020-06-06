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
from rlil.diag_q.pendulum_util import *
matplotlib.use('Agg')


def plot_visitation(fig, visitation, row_id, num_rows, num_cols=5,
                    start_step=0, end_step=MAX_STEPS,
                    vmin=None, vmax=None):
    vmin = visitation[start_step:end_step].min() if vmin is None else vmin
    vmax = visitation[start_step:end_step].max() if vmax is None else vmax
    step_interval = (end_step - start_step) / num_cols

    th_ticks, thv_ticks = [], []
    for i in range(STATE_DISC):
        th, thv = ENV.wrapped_env.from_disced_th_thv(i, i)
        th_ticks.append(round(th, 3))
        thv_ticks.append(round(thv, 3))

    # plot step == start_step
    data = pd.DataFrame(
        visitation[start_step, :, :], index=th_ticks,
        columns=thv_ticks).rolling(3).max().rolling(3, axis=1).max()
    ax = fig.add_subplot(num_rows, num_cols, row_id * num_cols + 1)
    sns.heatmap(data, ax=ax, cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title("time step: {}".format(start_step))
    ax.set_ylabel(r"$\theta$")
    ax.set_xlabel(r"$\dot{\theta}$")

    # plot step == start_step + step_interval ~ end_step - 1
    for i in range(1, num_cols-1):
        idx = int(i * step_interval) + start_step
        data = pd.DataFrame(
            visitation[idx, :, :], index=th_ticks,
            columns=thv_ticks).rolling(3).max().rolling(3, axis=1).max()
        ax = fig.add_subplot(num_rows, num_cols, row_id * num_cols + i + 1)
        sns.heatmap(data, ax=ax, cbar=False,
                    yticklabels=False, vmin=vmin, vmax=vmax)
        ax.set_title("time step: {}".format(idx))
        ax.set_xlabel(r"$\dot{\theta}$")

    # plot step == end_step - 1
    data = pd.DataFrame(
        visitation[-1, :, :], index=th_ticks,
        columns=thv_ticks).rolling(3).max().rolling(3, axis=1).max()
    ax = fig.add_subplot(num_rows, num_cols, (row_id + 1) * num_cols)
    sns.heatmap(data, ax=ax,
                yticklabels=False, vmin=vmin, vmax=vmax)
    ax.set_title("time step: {}".format(end_step-1))
    ax.set_xlabel(r"$\dot{\theta}$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qval_path", type=str, default="./pendulum_q.npy")
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=MAX_STEPS)
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
        figure_title += runs_dir.name + "_visit_{}-{}.png".format(args.start_step, args.end_step)
        agent_dirs = sorted([agent_dir for agent_dir in runs_dir.iterdir()
                             if agent_dir.is_dir()])
    figure_title += "visit_{}-{}.png".format(args.start_step, args.end_step)

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

    # plot visitation
    print("Plotting true policy visitation ...")
    true_q_vals = compute_true_action_values(args.qval_path)
    true_visitation = sample_visitation(true_q_vals, sample_iters=1000)

    big_axes[0].set_title("True visitation map\n\n", fontsize="25")
    plot_visitation(fig, true_visitation,
                    row_id=0,
                    num_rows=num_rows,
                    num_cols=args.num_cols,
                    start_step=args.start_step,
                    end_step=args.end_step)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_title))
