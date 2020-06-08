from torch.optim import Adam
from rlil.agents import DQN
from rlil.approximation import QNetwork, FixedTarget
from rlil.memory import ExperienceReplayBuffer
from rlil.utils.scheduler import LinearScheduler
from rlil.policies import GreedyPolicy
from rlil.initializer import (get_device,
                              set_replay_buffer,
                              disable_on_policy_mode,
                              set_n_step,
                              enable_apex)
from .models import fc_relu_q


def dqn(
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
    DQN classic control preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        prioritized (bool): Use prioritized experience replay if True.
        use_apex (bool): Use apex if True.
        n_step (int): Number of steps for N step experience replay.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The training step where the exploration decay stops.
    """
    def _dqn(env):
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

        return DQN(
            q,
            policy,
            epsilon=epsilon,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
        )
    return _dqn


__all__ = ["dqn"]
