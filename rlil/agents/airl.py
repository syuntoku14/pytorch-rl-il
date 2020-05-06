import torch
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil import nn
from .gail import GAIL


class AIRL(GAIL):
    """
    Adversarial inverse reinforcement learning (AIRL)

    AIRL is an inverse reinforcement learning algorithm based on 
    adversarial learning. AIRL trains not only the reward function
    but also the value function to make the reward function robust 
    to changes in dynamics.

    Args:
        base_agent (rlil.agent.Agent): 
            An off-policy agent such as ddpg, td3, sac
        minibatch_size (int): 
            The number of experiences to sample in each discriminator update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of base_agent update per discriminator update
    """

    def __init__(self,
                 base_agent,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=10,
                 ):
        # objects
        self.base_agent = base_agent
        self.replay_buffer = get_replay_buffer()
        self.reward_fn = self.replay_buffer.reward_fn
        self.value_fn = self.replay_buffer.value_fn
        self.writer = get_writer()
        self.device = get_device()
        self.discrim_criterion = nn.BCELoss()
        # hyperparameters
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self._train_count = 0

    def train(self):
        # train discriminator
        if self.should_train():
            samples, expert_samples = self.replay_buffer.sample_both(
                self.minibatch_size)
            states, actions, _, next_states, _ = samples
            exp_states, exp_actions, _, exp_next_states, _ = expert_samples

            fake = self.replay_buffer.discrim(states, actions, next_states)
            real = self.replay_buffer.discrim(exp_states,
                                              exp_actions,
                                              exp_next_states)
            discrim_loss = self.discrim_criterion(fake, torch.ones_like(fake)) + \
                self.discrim_criterion(real, torch.zeros_like(real))
            discrim_loss.backward()
            self.reward_fn.reinforce(discrim_loss, backward=False)
            self.value_fn.reinforce(discrim_loss, backward=False)

            # additional debugging info
            self.writer.add_scalar('airl/fake', fake.mean())
            self.writer.add_scalar('airl/real', real.mean())

        # train base_agent
        self.base_agent.train()
