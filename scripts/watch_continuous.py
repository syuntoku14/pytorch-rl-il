# pylint: disable=unused-import
import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment
from rlil.experiments import GreedyAgent, watch
from continuous import ENVS
import logging
logging.basicConfig(level=logging.DEBUG)

def watch_continuous():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
    parser.add_argument("env", help="ID of the Environment")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=120,
        help="Playback speed",
    )
    args = parser.parse_args()

    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id)
    agent = GreedyAgent.load(args.dir, env, device=args.device)
    watch(agent, env, fps=args.fps, dir=args.dir)

if __name__ == "__main__":
    watch_continuous()
