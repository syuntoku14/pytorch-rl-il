import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")


def get_results(exp_path):

    def read_scalars(resultpath):
        # read scalars from event file
        for p in resultpath.rglob("events*"):
            eventspath = p
        event_acc = event_accumulator.EventAccumulator(
            str(eventspath), size_guidance={'scalars': 0})
        event_acc.Reload()

        scalars = {}
        steps = {}

        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            scalars[tag] = [event.value for event in events]
            steps[tag] = [event.step for event in events]

        return steps, scalars

    def get_return_dataframe(steps, scalars):
        # convert steps and scalars to dataframe
        df_dict = {}
        for key in steps.keys():
            step = key.split("/")[-1]
            tag = key.split("/")[-2]
            if tag != "returns":
                continue

            if "frame" in step:
                dicimal = -6
                df_dict[step] = pd.DataFrame(data={"samples": np.round(steps[key], dicimal),
                                                   "return": scalars[key]})
            elif "episode" in step:
                dicimal = -3
                df_dict[step] = pd.DataFrame(data={"episodes": np.round(steps[key], dicimal),
                                                   "return": scalars[key]})

        return pd.concat(df_dict, axis=1)

    results = defaultdict(lambda: defaultdict(lambda: []))
    for env in exp_path.glob("[!.]*[!.png]"):
        for result in env.glob("[!.]*"):
            agent = result.name.split("_")[1]
            steps, scalars = read_scalars(result)
            df = get_return_dataframe(steps, scalars)
            results[env.name][agent].append(df)

        # concatenate same agent
        for agent in results[env.name]:
            results[env.name][agent] = \
                pd.concat(results[env.name][agent])

    return results


def plot(exp_path, step="sample_frame", xlim=None):
    exp_path = Path(exp_path)
    results = get_results(exp_path)

    # layout
    if "frame" in step:
        x = "samples"
    elif "episode" in step:
        x = "episodes"
    num_cols = len(results)
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols*6, 4))
    if num_cols == 1:
        axes = [axes]
    sns.set(style="darkgrid")

    # agent colors
    colors = sns.color_palette()
    agents = []

    for i, env in enumerate(results):
        for agent in results[env]:
            df = results[env][agent][step]
            if agent not in agents:
                agents.append(agent)

            sns.lineplot(x=x,
                         y="return",
                         ci="sd",
                         data=df,
                         ax=axes[i],
                         label=agent,
                         legend=None,
                         color=colors[agents.index(agent)])

            axes[i].set_title(env)
            axes[i].set_xlim(0, xlim)

    handles = [None] * len(agents)

    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        for h, agent in zip(handle, label):
            handles[agents.index(agent)] = h

    lgd = fig.legend(handles, agents, loc="upper center",
                     bbox_to_anchor=(0.5, 1.1), ncol=len(agents))
    fig.tight_layout()
    fig.savefig(str(exp_path / "result.png"),
                bbox_extra_artists=(lgd, ), bbox_inches="tight")
