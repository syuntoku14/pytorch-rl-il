from torch.optim import Adam
from rlil.approximation import QNetwork, FixedTarget
from rlil.memory import ExperienceReplayBuffer
from rlil.utils.scheduler import LinearScheduler
from rlil.policies import GreedyPolicy
from rlil.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode,
                              set_n_step,
                              enable_apex)
from rlil.diag_q.agents import GridCraftDQN
from .models import fc_relu_q


def gridcraft_dqn(
        # Common settings
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-3,
        # Training settings
        minibatch_size=100,
        target_update_frequency=100,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=1e7,
        prioritized=False,
        use_apex=False,
        n_step=1,
        # Exploration settings
        initial_exploration=1.,
        final_exploration=0.,
        final_exploration_step=10000,
):
    """
    DQN preset for gridcraft.
    """
    def _gridcraft_dqn(env):
        disable_on_policy_mode()

        device = get_device()
        model = fc_relu_q(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(
            model,
            optimizer,
            target=FixedTarget(target_update_frequency),
        )
        policy = GreedyPolicy(
            model,
            env.action_space.n,
            epsilon=initial_exploration
        )
        epsilon = LinearScheduler(
            initial_exploration,
            final_exploration,
            final_exploration_step,
            name="epsilon"
        )

        if use_apex:
            enable_apex()
        set_n_step(n_step=n_step, discount_factor=discount_factor)
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size, env,
            prioritized=prioritized or use_apex)
        set_replay_buffer(replay_buffer)

        return GridCraftDQN(
            env.env.env.wrapped_env,
            q,
            policy,
            epsilon=epsilon,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
        )
    return _gridcraft_dqn


__all__ = ["gridcraft_dqn"]
