import torch
from torch.optim import Adam
from rlil.agents import RsMPC
from rlil.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode)
from rlil.approximation import Dynamics
from rlil.memory import ExperienceReplayBuffer
from rlil.environments import REWARDS
from .models import fc_dynamics


def rs_mpc(
        horizon=20,
        num_samples=1000,
        # Adam optimizer settings
        lr_dyn=1e-3,
        # Training settings
        minibatch_size=100,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e7,
):
    """
    Rnadom shooting MPC (RsMPC) control preset

    Args:
        horizon (int): Control horizon.
        num_samples (int): Number of action samples for random shooting.
        lr_dyn (float): Learning rate for the dynamics network.
        minibatch_size (int): Number of experiences to sample in each training update.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """
    def _rs_mpc(env):
        assert env.name in REWARDS, \
            "The reward function of {} is not registered in rlil.environments.reward_fns."
        reward_fn = REWARDS[env.name]()

        disable_on_policy_mode()
        device = get_device()

        dynamics_model = fc_dynamics(env).to(device)
        dynamics_optimizer = Adam(dynamics_model.parameters(), lr=lr_dyn)
        dynamics = Dynamics(
            dynamics_model,
            dynamics_optimizer,
        )

        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, env)
        set_replay_buffer(replay_buffer)

        return RsMPC(
            dynamics=dynamics,
            reward_fn=reward_fn,
            horizon=horizon,
            num_samples=num_samples,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size
        )
    return _rs_mpc


__all__ = ["rs_mpc"]
