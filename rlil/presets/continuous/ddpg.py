import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from rlil.agents import DDPG
from rlil.approximation import QContinuous, PolyakTarget
from rlil.policies import DeterministicPolicy
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import get_device, set_replay_buffer
from .models import fc_q, fc_deterministic_policy


def ddpg(
        # pretrained policy path
        policy_path=None,
        # Common settings
        discount_factor=0.99,
        last_frame=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
        update_frequency=1,
        polyak_rate=0.005,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6,
        # Exploration settings
        noise=0.1,
):
    """
    DDPG continuous control preset.

    Args:
        policy_path (str): Path to the pretrained policy state_dict.pt
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        noise (float): The amount of exploration noise to add.
    """
    def _ddpg(env):
        final_anneal_step = (
            last_frame - replay_start_size) // update_frequency

        device = get_device()
        q_model = fc_q(env).to(device)
        q_optimizer = Adam(q_model.parameters(), lr=lr_q)
        q = QContinuous(
            q_model,
            q_optimizer,
            target=PolyakTarget(polyak_rate),
            lr_scheduler=CosineAnnealingLR(
                q_optimizer,
                final_anneal_step
            ),
        )

        policy_model = fc_deterministic_policy(env).to(device)
        if policy_path:
            policy_model.load_state_dict(
                torch.load(policy_path, map_location=device))
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
            lr_scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size
        )

        set_replay_buffer(replay_buffer)

        return DDPG(
            q,
            policy,
            noise=noise,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
        )
    return _ddpg


__all__ = ["ddpg"]
