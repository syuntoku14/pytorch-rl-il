import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment, ENVS
from rlil.experiments import Experiment
from rlil.presets import continuous, get_default_args
from rlil.initializer import get_logger, set_device
import torch
import logging
import ray


def main():
    parser = argparse.ArgumentParser(
        description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (see envs)")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. actor_critic). See presets for available agents.",
                        )
    parser.add_argument("--frames", type=int, default=5e7,
                        help="The number of training frames")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
                        )
    parser.add_argument("--render", default=False,
                        help="Whether to render the environment.")
    parser.add_argument("--policy", default=None,
                        help="Path to the pretrained policy state_dict")
    parser.add_argument("--exp_info", default="default experiment",
                        help="One line descriptions of the experiment. Experiments' results are saved in 'runs/[exp_info]/[env_id]/'"
                        )

    args = parser.parse_args()

    ray.init(include_webui=False, ignore_reinit_error=True)

    set_device(torch.device(args.device))

    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id)
    agent_name = args.agent
    preset = getattr(continuous, agent_name)
    preset_args = get_default_args(preset)
    agent_fn = preset(policy_path=args.policy)

    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    args_dict = vars(args)
    args_dict.update(preset_args)

    Experiment(
        agent_fn, env, num_workers=args.num_workers, max_frames=args.frames, 
        args_dict=args_dict, exp_info=args.exp_info
    )


if __name__ == "__main__":
    main()
