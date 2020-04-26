import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment, ENVS
from rlil.experiments import Experiment
from rlil.presets import get_default_args
from rlil.presets.gail import continuous
import rlil.presets.online.continuous as online_continuous
from rlil.initializer import get_logger, set_device, set_seed
import torch
import logging
import ray
import pickle
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run a gail benchmark.")
    parser.add_argument("env", help="Name of the env")
    parser.add_argument("gail_agent",
                        help="Name of the gail agent (e.g. gail). \
                            See presets for available agents.")
    parser.add_argument("base_agent",
                        help="Name of the base agent (e.g. ddpg) for gail. \
                            See presets for available agents.")
    parser.add_argument("dir",
                        help="Directory where the transitions.pkl is saved.")
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--train_steps", type=int, default=1e7,
                        help="Number of training steps")
    parser.add_argument("--num_workers", type=int,
                        default=1, help="Number of workers for training")
    parser.add_argument("--exp_info", default="default experiment",
                        help="One line descriptions of the experiment. \
                            Experiments' results are saved in 'runs/[exp_info]/[env_id]/'")

    args = parser.parse_args()

    # initialization
    ray.init(include_webui=False, ignore_reinit_error=True)
    set_device(torch.device(args.device))
    set_seed(args.seed)
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # set environment
    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env
    env = GymEnvironment(env_id)

    # set base_agent
    base_preset = getattr(online_continuous, args.base_agent)
    base_agent_fn = base_preset()

    # set gail_agent
    with open(os.path.join(args.dir + "transitions.pkl"), mode='rb') as f:
        transitions = pickle.load(f)
    gail_preset = getattr(continuous, args.gail_agent)
    agent_fn = gail_preset(
        transitions=transitions,
        base_agent_fn=base_agent_fn,
    )

    agent_name = agent_fn.__name__[1:] + "-" + \
        base_agent_fn.__name__[1:]

    # set args_dict
    args_dict = {"args": {}, "base": {}, "gail": {}}
    args_dict["args"] = vars(args)
    args_dict["base"] = get_default_args(base_preset)
    args_dict["gail"] = get_default_args(gail_preset)

    Experiment(
        agent_fn, env,
        agent_name=agent_name,
        num_workers=args.num_workers,
        max_train_steps=args.train_steps,
        args_dict=args_dict,
        exp_info=args.exp_info,
    )


if __name__ == "__main__":
    main()
