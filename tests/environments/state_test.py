import pytest
import numpy as np
import torch
import torch_testing as tt
from rlil.environments.state import State, DONE, NOT_DONE


def test_constructor_defaults():
    raw = torch.randn(3, 4)
    state = State(raw)
    # state.features returns raw
    tt.assert_equal(state.features, raw)
    # state.mask returns 1 default
    tt.assert_equal(state.mask, torch.ones(3, dtype=torch.bool))
    # state.raw == raw
    tt.assert_equal(state.raw, raw)
    assert state.info == [None] * 3


def test_custom_constructor_args():
    raw = torch.randn(3, 4)
    mask = torch.zeros(3).bool()
    info = ['a', 'b', 'c']
    state = State(raw, mask=mask, info=info)
    tt.assert_equal(state.features, raw)
    # check zeros masks
    tt.assert_equal(state.mask, torch.zeros(3, dtype=torch.bool))
    # check info constructor
    assert state.info == info


def test_not_done():
    state = State(torch.randn(1, 4))
    assert not state.done


def test_done():
    raw = torch.randn(1, 4)
    state = State(raw, mask=DONE)
    assert state.done


def test_from_list():
    state1 = State(torch.randn(1, 4), mask=DONE, info=['a'])
    state2 = State(torch.randn(1, 4), mask=NOT_DONE, info=['b'])
    state3 = State(torch.randn(1, 4))
    state = State.from_list([state1, state2, state3])
    tt.assert_equal(state.raw, torch.cat(
        (state1.raw, state2.raw, state3.raw)))
    tt.assert_equal(state.mask, torch.tensor([0, 1, 1]))
    assert state.info == ['a', 'b', None]


def test_from_gym():
    gym_obs = np.array([1, 2, 3])
    done = True
    info = 'a'
    state = State.from_gym(gym_obs, done, info)
    tt.assert_equal(state.raw, torch.tensor([[1, 2, 3]]))
    tt.assert_equal(state.mask, DONE)
    assert state.info == ['a']


def test_get_item():
    raw = torch.randn(3, 4)
    states = State(raw)
    state = states[2]
    tt.assert_equal(state.raw, raw[2].unsqueeze(0))
    tt.assert_equal(state.mask, NOT_DONE)
    assert state.info == [None]


def test_len():
    state = State(torch.randn(3, 4))
    assert len(state) == 3
