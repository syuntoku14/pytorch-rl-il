import argparse
from rlil.utils.plots import plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the results of experiments.")
    parser.add_argument("dir",
                        help="Experiment directory. This is a directory of exp_info, not runs/")
    parser.add_argument("--step", type=str, default="train_step",
                        help="The unit of x-axis. You can choose it from [sample_frame, sample_episode, train_step]")
    parser.add_argument("--xlim", type=int, default=None, help="The limit of x-axis.")
 
    args = parser.parse_args()

    plot(args.dir, args.step, args.xlim)