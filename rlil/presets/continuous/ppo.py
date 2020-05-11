from torch.optim import Adam
from rlil.agents import PPO
from rlil.approximation import VNetwork, FeatureNetwork, Approximation
from rlil.policies import GaussianPolicy
from rlil.memory import ExperienceReplayBuffer, GaeWrapper
from rlil.initializer import (get_writer,
                              get_device,
                              set_replay_buffer,
                              enable_on_policy_mode)
from .models import fc_actor_critic


def ppo(
        # Common settings
        discount_factor=0.98,
        # Adam optimizer settings
        lr=3e-4,  # Adam learning rate
        eps=1e-5,  # Adam stability
        # Loss scaling
        entropy_loss_scaling=0.0,
        value_loss_scaling=0.5,
        # Replay Buffer settings
        replay_start_size=5000,
        # Training settings
        clip_grad=0.5,
        epsilon=0.2,
        minibatches=4,
        epochs=2,
        # GAE settings
        lam=0.95,
):
    """
    PPO continuous control preset.

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
        epsilon (float): 
            Epsilon value in the clipped PPO objective function.
        minibatches (int): The number of minibatches to split each batch into.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
    """
    def _ppo(env):
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
        replay_buffer = GaeWrapper(replay_buffer, discount_factor, lam)
        set_replay_buffer(replay_buffer)

        return PPO(
            feature_nw,
            v,
            policy,
            epsilon=epsilon,
            replay_start_size=replay_start_size,
            minibatches=minibatches,
            entropy_loss_scaling=entropy_loss_scaling,
        )

    return _ppo


__all__ = ["ppo"]
