import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from rlil.memory import set_replay_buffer
from rlil.agents import BCQ
from rlil.approximation import QContinuous, PolyakTarget, AutoEncoder
from rlil.policies import BCQDeterministicPolicy
from rlil.memory import ExperienceReplayBuffer
from rlil.utils import get_device
from .models import fc_q, fc_bcq_deterministic_policy, \
    FC_Encoder_BCQ, FC_Decoder_BCQ


def bcq(
        replay_buffer,
        # pretrained policy path
        policy_path=None,
        # Common settings
        discount_factor=0.98,
        last_frame=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        lr_vae=1e-3,
        # Training settings
        minibatch_size=100,
        polyak_rate=0.005,
        noise_td3=0.2,
        policy_update_td3=2,
        # Exploration settings
        noise_policy=0.1,
):
    """
    Batch-Constrained Q-learning (BCQ) control preset

    Args:
        replay_buffer (ExperienceReplayBuffer): ExperienceReplayBuffer with expert trajectory
        policy_path (str): Path to the pretrained policy state_dict.pt
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        lr_vae (float): Learning rate for the VAE.
        minibatch_size (int): Number of experiences to sample in each training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        noise_td3 (float): the amount of noise to add to each action in trick three.
        policy_update_td3 (int): Number of timesteps per training update the policy in trick two.
        noise_policy (float): The amount of exploration noise to add.
    """
    def _bcq(env):
        final_anneal_step = last_frame 

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

        policy_model = fc_bcq_deterministic_policy(env).to(device)
        if policy_path:
            policy_model.load_state_dict(
                torch.load(policy_path, map_location=device))
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = BCQDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
            lr_scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
        )

        encoder_model = FC_Encoder_BCQ(env).to(device)
        decoder_model = FC_Decoder_BCQ(env).to(device)
        model_parameters = (list(encoder_model.parameters()) +
                            list(decoder_model.parameters()))
        vae_optimizer = Adam(model_parameters, lr=lr_vae)
        vae = AutoEncoder(
            encoder_model,
            decoder_model,
            vae_optimizer,
            lr_scheduler=CosineAnnealingLR(
                vae_optimizer,
                final_anneal_step
            ),
            name="VAE",
            )

        set_replay_buffer(replay_buffer)

        return BCQ(
            q_1,
            q_2,
            vae,
            policy,
            noise_policy=noise_policy,
            noise_td3=noise_td3,
            policy_update_td3=policy_update_td3,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
        )
    return _bcq


__all__ = ["bcq"]
