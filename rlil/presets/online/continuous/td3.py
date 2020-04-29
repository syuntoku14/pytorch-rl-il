import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from rlil.agents import TD3
from rlil.approximation import QContinuous, PolyakTarget
from rlil.policies import DeterministicPolicy
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode)
from .models import fc_q, fc_deterministic_policy


def td3(
        # Common settings
        discount_factor=0.99,
        last_step=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
        polyak_rate=0.005,
        noise_td3=0.2,
        policy_update_td3=2,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6,
        # Exploration settings
        noise_policy=0.1,
):
    """
    TD3 continuous control preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        last_step (int): Number of steps to train.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        noise_td3 (float): the amount of noise to add to each action in trick three.
        policy_update_td3 (int): Number of timesteps per training update the policy in trick two.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        noise_policy (float): The amount of exploration noise to add.
    """
    def _td3(env):
        disable_on_policy_mode()
        final_anneal_step = (last_step - replay_start_size)

        device = get_device()
        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            target=PolyakTarget(polyak_rate),
            lr_scheduler=CosineAnnealingLR(
                q_1_optimizer,
                final_anneal_step
            ),
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            target=PolyakTarget(polyak_rate),
            lr_scheduler=CosineAnnealingLR(
                q_2_optimizer,
                final_anneal_step
            ),
            name='q_2'
        )

        policy_model = fc_deterministic_policy(env).to(device)
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

        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, env)
        set_replay_buffer(replay_buffer)

        return TD3(
            q_1,
            q_2,
            policy,
            noise_policy=noise_policy,
            noise_td3=noise_td3,
            policy_update_td3=policy_update_td3,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size
        )
    return _td3


__all__ = ["td3"]
