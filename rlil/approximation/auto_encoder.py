import os
import torch
from torch.nn import utils
from rlil.utils.writer import DummyWriter
from rlil.nn import RLNetwork
from .approximation import DEFAULT_CHECKPOINT_FREQUENCY
from .checkpointer import PeriodicCheckpointer


class AutoEncoder:
    def __init__(
            self,
            encoder_model,
            decoder_moder,
            optimizer,
            encoder_checkpointer=None,
            decoder_checkpointer=None,
            clip_grad=0,
            loss_scaling=1,
            name="AE",
            lr_scheduler=None,
            writer=DummyWriter(),
    ):
        self.encoder_model = encoder_model
        self.decoder_moder = decoder_moder
        self.device = next(encoder_model.parameters()).device
        self._lr_scheduler = lr_scheduler
        self._optimizer = optimizer
        self._loss_scaling = loss_scaling
        self._clip_grad = clip_grad
        self._writer = writer
        self._name = name
        if encoder_checkpointer is None:
            encoder_checkpointer = PeriodicCheckpointer(
                DEFAULT_CHECKPOINT_FREQUENCY)
        if decoder_checkpointer is None:
            decoder_checkpointer = PeriodicCheckpointer(
                DEFAULT_CHECKPOINT_FREQUENCY)
        self._encoder_checkpointer = encoder_checkpointer
        self._decoder_checkpointer = decoder_checkpointer
        self._encoder_checkpointer.init(
            self.encoder_model,
            os.path.join(writer.log_dir, name + '_encoder.pt')
        )
        self._decoder_checkpointer.init(
            self.decoder_moder,
            os.path.join(writer.log_dir, name + '_decoder.pt')
        )

    def encode(self, *inputs):
        return self.encoder_model(*inputs)

    def decode(self, *inputs):
        return self.decoder_moder(*inputs)

    def reinforce(self, loss):
        loss = self._loss_scaling * loss
        self._writer.add_loss(self._name, loss.detach())
        loss.backward()
        self.step()
        return self

    def step(self):
        '''Given that a backward pass has been made, run an optimization step.'''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(
                self.encoder_model.parameters(), self._clip_grad)
            utils.clip_grad_norm_(
                self.decoder_moder.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        if self._lr_scheduler:
            self._writer.add_schedule(
                self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._lr_scheduler.step()
        self._encoder_checkpointer()
        self._decoder_checkpointer()
        return self

    def zero_grad(self):
        self._optimizer.zero_grad()
        return self
