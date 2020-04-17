import torch
from torch.optim import Adam
from rlil.agents import BC
from rlil.initializer import get_device, set_replay_buffer
from rlil.policies import DeterministicPolicy
from .models import fc_deterministic_policy


def bc(
        replay_buffer,
        # Adam optimizer settings
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
):
    """
    Behavioral Cloning (BC) control preset

    Args:
        replay_buffer (ExperienceReplayBuffer): ExperienceReplayBuffer with expert trajectory
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
    """
    def _bc(env):
        device = get_device()

        policy_model = fc_deterministic_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
        )

        set_replay_buffer(replay_buffer)

        return BC(
            policy=policy,
            minibatch_size=minibatch_size,
        )
    return _bc


__all__ = ["bc"]
