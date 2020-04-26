import torch
from rlil.initializer import get_device, get_writer, get_replay_buffer
from rlil import nn
from .base import Agent


class GAIL(Agent):
    """
    Generative adversarial imitation learning (GAIL)

    GAIL is composed of two neural networks, the policy (generator) network 
    and the discriminator network. In the original paper (https://arxiv.org/abs/1606.03476),
    the policy network is trained using TRPO.
    Since GAIL can be applicable to any policy update algorithms, 
    the implementation in rlil is combined with off-policy algorithms,
    such as ddpg or td3.

    Args:
        base_agent (rlil.agent.Agent): 
            An off-policy agent such as ddpg, td3, sac
        minibatch_size (int): 
            The number of experiences to sample in each discriminator update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
    """

    def __init__(self,
                 base_agent,
                 minibatch_size=32,
                 replay_start_size=5000,
                 ):
        # objects
        self.base_agent = base_agent
        self.replay_buffer = get_replay_buffer()
        self.discriminator = self.replay_buffer.discriminator
        self.writer = get_writer()
        self.device = get_device()
        self.discrim_criterion = nn.BCELoss()
        # hyperparameters
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        # private
        self._train_count = 0

    def act(self, *args, **kwargs):
        return self.base_agent.act(*args, **kwargs)

    def train(self):
        # train discriminator
        if self._should_train():
            samples, expert_samples = self.replay_buffer.sample_both(
                self.minibatch_size)
            states, actions, _, _, _ = samples
            exp_samples, exp_actions, _, _, _ = expert_samples

            fake = self.discriminator(states, actions)
            real = self.discriminator(exp_states, exp_actions)
            discrim_loss = self.discrim_criterion(fake, torch.ones_like(fake)) + \
                self.discrim_criterion(real, torch.zeros_like(real))
            self.discriminator.reinforce(discrim_loss)

        # train base_agent
        self.base_agent.train()

    def _should_train(self):
        self._train_count += 1
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, *args, **kwargs):
        return self.base_agent.make_lazy_agent(*args, **kwargs)

    def load(self, dirname):
        self.agent.load(dirname)
