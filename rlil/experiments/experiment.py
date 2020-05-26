import numpy as np
from rlil.utils.writer import ExperimentWriter
from rlil.initializer import get_logger, get_writer, set_writer, set_logger, set_seed
from rlil.samplers import AsyncSampler
from .trainer import Trainer
import os
import logging
import json
import git
import warnings


class Experiment:
    def __init__(
            self,
            agent_fn,
            env,
            agent_name=None,
            args_dict={},
            exp_info="default_experiments",
            seed=0,
            trains_per_episode=20,
            num_workers=1,
            num_workers_eval=1,
            max_sample_frames=np.inf,
            max_sample_episodes=np.inf,
            max_train_steps=np.inf,
            train_minutes=np.inf
    ):
        # set_seed
        set_seed(seed)

        # set writer
        if agent_name is None:
            agent_name = agent_fn.__name__[1:].replace("_", "-")
        writer = self._make_writer(agent_name, env.name, exp_info)
        message = "\n# Experiment: " + exp_info
        message += "  \n# Parameters:  \n"
        message += json.dumps(args_dict, indent=4,
                              sort_keys=True).replace("\n", "  \n")

        # write git diff
        try:
            repo = git.Repo('./')
            t = repo.head.commit.tree
            diff = repo.git.diff(t).replace("\n", "  \n")
            message += "  \n# Git diff:  \n" + diff
        except git.InvalidGitRepositoryError:
            warnings.warn(
                "Current repository doesn't have .git. git diff is not recorded.")

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

        sampler = AsyncSampler(env, num_workers=num_workers) \
            if num_workers > 0 else None
        eval_sampler = AsyncSampler(env, num_workers=num_workers_eval) \
            if num_workers_eval > 0 else None

        trainer = Trainer(
            agent=agent,
            sampler=sampler,
            eval_sampler=eval_sampler,
            trains_per_episode=trains_per_episode,
            max_sample_frames=max_sample_frames,
            max_sample_episodes=max_sample_episodes,
            max_train_steps=max_train_steps,
            train_minutes=train_minutes
        )

        trainer.start_training()

    def _make_writer(self, agent_name, env_name, exp_info):
        return ExperimentWriter(agent_name=agent_name,
                                env_name=env_name,
                                exp_info=exp_info)
