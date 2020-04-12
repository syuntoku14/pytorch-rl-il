import argparse
from rlil.utils.plots import plot_returns_100


def plot():
    parser = argparse.ArgumentParser(
        description="Plots the results of experiments.")
    parser.add_argument("dir", help="Output directory.")
    parser.add_argument("--smooth", type=int, default=11,
                        help="The window size of the moving average")
    parser.add_argument("--timesteps", type=int, default=-1,
                        help="The final point will be fixed to this x-value")
    args = parser.parse_args()
    plot_returns_100(args.dir, smooth=args.smooth, timesteps=args.timesteps)


if __name__ == "__main__":
    plot()
