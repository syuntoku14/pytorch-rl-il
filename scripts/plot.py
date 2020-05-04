import argparse
from rlil.utils.plots import plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the results of experiments.")
    parser.add_argument("dir",
                        help="Experiment directory. This is a directory of exp_info, not runs/")
    parser.add_argument("--step", type=str, default="train_steps",
                        help="The unit of x-axis. You can choose it from \
                            [sample_frames, sample_episodes, train_steps, minutes]")

    args = parser.parse_args()

    plot(args.dir, args.step)
