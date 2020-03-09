import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from rlil.agents import TD3
from rlil.approximation import QContinuous, PolyakTarget
from rlil.utils.writer import DummyWriter
from rlil.policies import DeterministicPolicy
from rlil.memory import ExperienceReplayBuffer
from .models import fc_q, fc_deterministic_policy


def td3(
        # pretrained policy path
        policy_path=None,
        # Common settings
        device="cpu",
        discount_factor=0.98,
        last_frame=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
        update_frequency=1,
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
        policy_path (str): Path to the pretrained policy state_dict.pt
        device (str): The device to load parameters and buffers onto for this agent..
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        noise_td3 (float): the amount of noise to add to each action in trick three.
        policy_update_td3 (int): Number of timesteps per training update the policy in trick two.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        noise_policy (float): The amount of exploration noise to add.
    """
    def _td3(env, writer=DummyWriter()):
        final_anneal_step = (last_frame - replay_start_size) // update_frequency

        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                q_1_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                q_2_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_2'
        )

        policy_model = fc_deterministic_policy(env).to(device)
        if policy_path:
            policy_model.load_state_dict(torch.load(policy_path, map_location=device))
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size
        )

        return TD3(
            q_1,
            q_2,
            policy,
            replay_buffer,
            env.action_space,
            noise_policy=noise_policy,
            noise_td3=noise_td3,
            policy_update_td3=policy_update_td3,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
            device=device
        )
    return _td3


__all__ = ["td3"]
