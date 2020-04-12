import numpy as np
from rlil.utils.writer import ExperimentWriter
from rlil.initializer import get_logger, get_writer, set_writer, set_logger, set_seed
from rlil.samplers import AsyncSampler
from .trainer import Trainer
import os
import logging
import json


class Experiment:
    def __init__(
            self,
            agent_fn,
            env,
            args_dict={},
            exp_info="default_experiments",
            seed=0,
            num_workers=1,
            num_workers_eval=1,
            max_frames=np.inf,
            max_episodes=np.inf,
    ):
        # set_seed
        seed = set_seed(seed)

        # set writer
        agent_name = agent_fn.__name__
        writer = self._make_writer(agent_name, env.name, exp_info)
        message = "# Parameters  \n"
        message += json.dumps(args_dict, indent=4,
                              sort_keys=True).replace("\n", "  \n")
        message += "  \n# Experiment infomation  \n" + exp_info
        writer.add_text("exp_summary", message)
        set_writer(writer)

        # set logger
        logger = get_logger()
        handler = logging.FileHandler(
            os.path.join(writer.log_dir, "logger.log"))
        fmt = logging.Formatter('%(levelname)s : %(asctime)s : %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        set_logger(logger)

        # save args
        with open(os.path.join(writer.log_dir, "args.json"), mode="w") as f:
            json.dump(args_dict, f)

        # start training
        agent = agent_fn(env)

        sampler = AsyncSampler(env, num_workers=num_workers)
        eval_sampler = AsyncSampler(env, num_workers=num_workers_eval)

        trainer = Trainer(
            agent,
            sampler,
            eval_sampler,
            max_frames,
            max_episodes
        )

        trainer.start_training()

    def _make_writer(self, agent_name, env_name, exp_info):
        return ExperimentWriter(agent_name=agent_name,
                                env_name=env_name,
                                exp_info=exp_info)
