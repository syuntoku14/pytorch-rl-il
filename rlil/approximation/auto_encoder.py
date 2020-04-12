import os
import torch
from torch.nn import utils
from rlil.initializer import get_writer
from rlil.nn import RLNetwork
from .approximation import DEFAULT_CHECKPOINT_FREQUENCY
from .checkpointer import PeriodicCheckpointer


class AutoEncoder:
    def __init__(
            self,
            encoder_model,
            decoder_model,
            optimizer,
            encoder_checkpointer=None,
            decoder_checkpointer=None,
            clip_grad=0,
            loss_scaling=1,
            name="AE",
            lr_scheduler=None,
    ):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.device = next(encoder_model.parameters()).device
        self._lr_scheduler = lr_scheduler
        self._optimizer = optimizer
        self._loss_scaling = loss_scaling
        self._clip_grad = clip_grad
        self._writer = get_writer()
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
            os.path.join(self._writer.log_dir, name + '_encoder.pt')
        )
        self._decoder_checkpointer.init(
            self.decoder_model,
            os.path.join(self._writer.log_dir, name + '_decoder.pt')
        )

    def encode(self, *inputs):
        return self.encoder_model(*inputs)

    def decode(self, *inputs):
        return self.decoder_model(*inputs)

    def reinforce(self, loss):
        loss = self._loss_scaling * loss
        self._writer.add_scalar("loss/" + self._name, loss.detach())
        loss.backward()
        self.step()
        return self

    def step(self):
        '''Given that a backward pass has been made, run an optimization step.'''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(
                self.encoder_model.parameters(), self._clip_grad)
            utils.clip_grad_norm_(
                self.decoder_model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._optimizer.zero_grad()
        if self._lr_scheduler:
            self._writer.add_schedule(
                "schedule" + self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._lr_scheduler.step()
        self._encoder_checkpointer()
        self._decoder_checkpointer()
        return self

    def zero_grad(self):
        self._optimizer.zero_grad()
        return self
