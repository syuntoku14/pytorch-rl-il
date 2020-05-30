import torch
from torch.optim import Adam
from rlil.agents import DDPG
from rlil.approximation import QContinuous, PolyakTarget
from rlil.policies import DeterministicPolicy
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode,
                              set_n_step,
                              enable_apex)
from .models import fc_q, fc_deterministic_policy


def ddpg(
        # Common settings
        discount_factor=0.99,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        # Training settings
        minibatch_size=512,
        polyak_rate=0.005,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e7,
        prioritized=False,
        use_apex=False,
        n_step=1,
        # Exploration settings
        noise=0.1,
):
    """
    DDPG continuous control preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        prioritized (bool): Use prioritized experience replay if True.
        use_apex (bool): Use apex if True.
        n_step (int): Number of steps for N step experience replay.
        noise (float): The amount of exploration noise to add.
    """
    def _ddpg(env):
        disable_on_policy_mode()

        device = get_device()
        q_model = fc_q(env).to(device)
        q_optimizer = Adam(q_model.parameters(), lr=lr_q)
        q = QContinuous(
            q_model,
            q_optimizer,
            target=PolyakTarget(polyak_rate),
        )

        policy_model = fc_deterministic_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
        )

        if use_apex:
            enable_apex()
        set_n_step(n_step=n_step, discount_factor=discount_factor)
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size, env,
            prioritized=prioritized or use_apex)
        set_replay_buffer(replay_buffer)

        return DDPG(
            q,
            policy,
            noise=noise,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
        )
    return _ddpg


__all__ = ["ddpg"]
