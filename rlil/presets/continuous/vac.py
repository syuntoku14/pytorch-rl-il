from torch.optim import Adam
from rlil.agents import VAC
from rlil.approximation import VNetwork, FeatureNetwork, Approximation
from rlil.policies import GaussianPolicy
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import (get_writer,
                              get_device,
                              set_replay_buffer,
                              enable_on_policy_mode)
from .models import fc_actor_critic


def vac(
        # Common settings
        discount_factor=0.98,
        # Adam optimizer settings
        lr=3e-4,  # Adam learning rate
        eps=1e-5,  # Adam stability
        # Loss scaling
        value_loss_scaling=0.5,
        # Replay Buffer settings
        replay_start_size=500,
        # Training settings
        clip_grad=0.5,
):
    """
    VAC continuous control preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        entropy_loss_scaling (float): 
            Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        clip_grad (float): 
            The maximum magnitude of the gradient for any given parameter. 
            Set to 0 to disable.
    """
    def _vac(env):
        enable_on_policy_mode()

        device = get_device()
        feature_model, value_model, policy_model = fc_actor_critic(env)
        feature_model.to(device)
        value_model.to(device)
        policy_model.to(device)

        feature_optimizer = Adam(
            feature_model.parameters(), lr=lr, eps=eps
        )
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)

        feature_nw = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
        )
        policy = GaussianPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            clip_grad=clip_grad,
        )

        replay_buffer = ExperienceReplayBuffer(1e7, env)
        set_replay_buffer(replay_buffer)

        return VAC(
            feature_nw,
            v,
            policy,
            discount_factor=discount_factor,
            replay_start_size=replay_start_size,
        )

    return _vac


__all__ = ["vac"]
