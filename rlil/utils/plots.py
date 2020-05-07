import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import json
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
            start_time = events[0].wall_time
            scalars[tag] = [event.value for event in events]
            steps[tag] = [event.step for event in events]
            # for training minutes steps
            min_tag = tag.split("/")[:-1] + ["minutes"]
            scalars["/".join(min_tag)] = [event.value for event in events]
            steps["/".join(min_tag)] = [(event.wall_time - start_time) / 60
                                        for event in events]

        return steps, scalars

    def get_return_dataframe(steps, scalars):
        # convert steps and scalars to dataframe
        df_dict = {}
        for key in steps.keys():
            step = key.split("/")[-1]
            tag = key.split("/")[-2]
            if tag != "returns":
                continue

            if "sample_frames" == step:
                dicimal = -5  # round 0.1M sample frames
                df_dict[step] = pd.DataFrame(
                    data={"samples": np.round(steps[key], dicimal),
                          "return": scalars[key]})
            elif "sample_episodes" == step:
                dicimal = -3  # round 1000 sample episodes
                df_dict[step] = pd.DataFrame(
                    data={"episodes": np.round(steps[key], dicimal),
                          "return": scalars[key]})
            elif "train_steps" == step:
                dicimal = -3  # round 1000 train steps
                df_dict[step] = pd.DataFrame(
                    data={"steps": np.round(steps[key], dicimal),
                          "return": scalars[key]})
            elif "minutes" == step:
                dicimal = -1  # round 10 minutes
                df_dict[step] = pd.DataFrame(
                    data={"minutes": np.round(steps[key], dicimal),
                          "return": scalars[key]})

        return pd.concat(df_dict, axis=1)

    def get_demo_return(resultpath):
        # load demo_return.json
        for p in resultpath.rglob("demo_return.json"):
            with open(str(p)) as f:
                demo_return = json.load(f)
                return demo_return["mean"]
        return None

    results = defaultdict(lambda: defaultdict(lambda: []))
    demo_returns = defaultdict(lambda: None)
    for env in exp_path.glob("[!.]*[!.png]"):
        for result in env.glob("[!.]*"):
            agent = result.name.split("_")[0]
            # load result
            steps, scalars = read_scalars(result)
            try:
                df = get_return_dataframe(steps, scalars)
            except ValueError:
                print(str(result) + " doesn't have data.")
                continue
            results[env.name][agent].append(df)

            # load demo_return
            demo_return = get_demo_return(result)
            if demo_return is not None:
                demo_returns[env.name] = demo_return

        # concatenate same agent
        for agent in results[env.name].keys():
            results[env.name][agent] = \
                pd.concat(results[env.name][agent])

    return results, demo_returns


def plot(exp_path, step="sample_frames"):
    exp_path = Path(exp_path)
    results, demo_returns = get_results(exp_path)

    # layout
    if "sample_frames" == step:
        x = "samples"
    elif "sample_episodes" == step:
        x = "episodes"
    elif "train_steps" == step:
        x = "steps"
    elif "minutes" == step:
        x = "minutes"
    else:
        raise ValueError("Invalid argument 'step'. step must be from\
            [sample_frames, sample_episodes, train_steps, minutes]")

    num_cols = len(results)
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols*6, 4))
    if num_cols == 1:
        axes = [axes]
    sns.set(style="darkgrid")

    # agent colors
    colors = sns.color_palette()
    agents = []

    for i, env in enumerate(results):
        xlim = 0
        for agent in results[env]:
            # plot results
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
            xlim = max(xlim, df[x].max())

        axes[i].set_title(env)
        axes[i].set_xlim(0, xlim)

        # plot demonstration
        demo_return = demo_returns[env]
        if demo_return is not None:
            if "Demonstration" not in agents:
                agents.append("Demonstration")
            axes[i].axhline(demo_return,
                            ls='--',
                            label="Demonstration",
                            color=colors[agents.index("Demonstration")])

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
