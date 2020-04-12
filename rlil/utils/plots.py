import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_returns_100(runs_dir, timesteps=-1, smooth=11):
    data = load_returns_100_data(runs_dir)
    num_exps = len(data.keys())
    num_envs = max([len(envs.keys()) for envs in data.values()])
    lines = {}
    fig, axes = plt.subplots(num_exps, num_envs)
    if num_exps * num_envs == 1:
        axes = [axes]

    for i, exp in enumerate(sorted(data.keys())):
        for j, env in enumerate(sorted(data[exp].keys())):
            env_data = data[exp][env]
            ax = axes[i][j]
            subplot_returns_100(ax, exp, env, env_data, lines,
                                smooth=smooth, timesteps=timesteps)
    fig.tight_layout()
    fig.legend(list(lines.values()), list(lines.keys()), loc="center right")
    plt.savefig(runs_dir + "/result.png")


def load_returns_100_data(runs_dir):
    data = defaultdict(lambda: defaultdict(lambda: {}))

    def add_data(exp_info, env, agent, file):
        data[exp_info][env][agent] = np.genfromtxt(
            file, delimiter=",").reshape((-1, 3))

    # list of experiments
    for exp_info in os.listdir(runs_dir):
        exp_info_path = os.path.join(runs_dir, exp_info)
        if os.path.isdir(exp_info_path):
            # list of environments
            for env in os.listdir(exp_info_path):
                env_path = os.path.join(exp_info_path, env)
                if os.path.isdir(env_path):
                    # list of agents
                    for agent_dir in os.listdir(env_path):
                        agent = agent_dir.split("_")[1].strip("_")
                        agent_path = os.path.join(env_path, agent_dir)
                        if os.path.isdir(agent_path):
                            returns100path = os.path.join(
                                agent_path, "returns100.csv")
                            # save data
                            if os.path.exists(returns100path):
                                add_data(exp_info, env, agent, returns100path)

    return data


def subplot_returns_100(ax, exp, env, data, lines, smooth=1, timesteps=-1):
    for agent in data:
        agent_data = data[agent]
        agent_data = agent_data[np.argsort(agent_data[:, 0])]
        end = agent_data[:, 0][-1] if timesteps < 0 else timesteps

        mean = agent_data[:, 1]
        std = agent_data[:, 2]
        if smooth > 1:
            y = np.ones(smooth)
            z = np.ones(len(mean))
            mean = np.convolve(mean, y, "same") / np.convolve(z, y, "same")
            std = np.convolve(std, y, "same") / np.convolve(z, y, "same")
        x = np.arange(0, 1e7, 1e4)
        mean = np.interp(x, agent_data[:, 0], mean)
        std = np.interp(x, agent_data[:, 0], std)

        if agent in lines:
            ax.plot(x, mean, label=agent, color=lines[agent].get_color())
        else:
            line, = ax.plot(x, mean, label=agent)
            lines[agent] = line
        ax.fill_between(
            x, mean + std, mean - std, alpha=0.2, color=lines[agent].get_color()
        )
        ax.set_title(env)
        ax.set_xlabel("timesteps")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 5))
