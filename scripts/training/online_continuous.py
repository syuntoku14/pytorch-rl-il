import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment, ENVS
from rlil.experiments import Experiment
from rlil.presets import get_default_args
from rlil.presets.online import continuous
from rlil.initializer import get_logger, set_device, set_seed
import torch
import logging
import ray


def main():
    parser = argparse.ArgumentParser(
        description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. actor_critic). See presets for available agents.")
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--train_minutes", type=int, default=60,
                        help="Minutes to train.")
    parser.add_argument("--trains_per_episode", type=int, default=20,
                        help="Number of training steps per episode")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for training")
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

    # set agent
    agent_name = args.agent
    preset = getattr(continuous, agent_name)
    agent_fn = preset()

    # set args_dict
    args_dict = get_default_args(preset)
    args_dict.update(vars(args))

    Experiment(
        agent_fn, env,
        num_workers=args.num_workers,
        train_minutes=args.train_minutes,
        args_dict=args_dict,
        seed=args.seed,
        exp_info=args.exp_info,
    )


if __name__ == "__main__":
    main()
