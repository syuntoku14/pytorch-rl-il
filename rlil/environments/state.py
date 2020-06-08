import numpy as np
import torch
from rlil.initializer import is_debug_mode, get_device
from copy import deepcopy


class State:
    def __init__(self, raw, mask=None, info=None):
        """
        Members of State object:
        1. raw (torch.Tensor): batch_size x shape
        2. mask (torch.BoolTensor): batch_size x 1
        3. info (list): batch_size
        """
        if is_debug_mode():
            # check if raw is valid
            if type(raw) is list:
                assert 5 > len(raw[0].shape) > 1, \
                    "State.raw[0].shape {} is invalid".format(raw[0].shape)
                assert len(raw[0].shape) > 1, \
                    "State.raw[0].shape {} is invalid. Batch_size must be specified".format(
                        raw[0].shape)
            else:
                assert isinstance(raw, torch.Tensor), \
                    "Input invalid raw type {}. raw must be torch.Tensor or list of torch.Tensor".format(
                        type(raw))
                assert 5 > len(raw.shape), \
                    "State.raw.shape {} is invalid".format(raw.shape)
                assert len(raw.shape) > 1, \
                    "State.raw.shape {} is invalid. Batch_size must be specified".format(
                        raw.shape)

            # check if info is valid
            if info is not None:
                assert type(info) == list, \
                    "info must be None or list"

            # check if mask is valid
            if mask is not None:
                assert isinstance(mask, torch.Tensor), \
                    "Input invalid mask type {}. mask must be torch.Tensor".format(
                        type(mask))
                assert len(mask.shape) == 1, \
                    "mask.shape {} must be 'shape == (batch_size)'".format(
                        mask.shape)
        self._raw = raw

        if mask is None:
            self._mask = torch.ones(
                len(raw),
                dtype=torch.bool,
                device=raw.device
            )
        else:
            self._mask = mask.bool()

        self._info = info or [None] * len(raw)

    def clone(self):
        return State(
            self._raw.clone(), self._mask.clone(), deepcopy(self._info)
        )

    @classmethod
    def from_list(cls, states):
        raw = torch.cat([state.features for state in states])
        done = torch.cat([state.mask for state in states])
        info = sum([state.info for state in states], [])
        return cls(raw, done, info)

    @classmethod
    def from_numpy(cls,
                   np_raw,
                   np_done=None,
                   info=None,
                   device="cpu"):
        raw = torch.as_tensor(np_raw, device=device)
        mask = ~torch.tensor(np_done, dtype=torch.bool,
                             device=device).reshape(-1) if np_done is not None else None
        info = info if info is not None else [None] * len(raw)
        return cls(raw, mask=mask, info=info)

    @property
    def features(self):
        '''
        Default features are the raw state.
        Override this method for other types of features.
        '''
        return self._raw

    @property
    def mask(self):
        return self._mask

    @property
    def info(self):
        return self._info

    @property
    def raw(self):
        return self._raw

    def raw_numpy(self):
        # return raw: np.array and done: np.array
        return self._raw.cpu().detach().numpy(), \
            self.done.cpu().detach().numpy()

    @property
    def done(self):
        return ~self._mask

    @property
    def device(self):
        return self._raw.device

    def to(self, device):
        return State(
            self._raw.to(device), self._mask.to(device), self._info
        )

    def detach(self):
        return State(
            self._raw.detach(), self._mask.detach(), self._info
        )

    @property
    def shape(self):
        return self._raw.shape

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return State(
                self._raw[idx],
                self._mask[idx],
                self._info[idx]
            )
        if isinstance(idx, torch.Tensor):
            return State(
                self._raw[idx],
                self._mask[idx],
                # can't copy info
            )
        return State(
            self._raw[idx].unsqueeze(0),
            self._mask[idx].unsqueeze(0),
            [self._info[idx]]
        )

    def __len__(self):
        return len(self._raw)
