import pytest
import numpy as np
from rlil.utils.writer import ExperimentWriter
from rlil.initializer import set_writer, get_writer
from shutil import rmtree
import pathlib
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


@pytest.fixture()
def init_writer():
    interval = {"sample_frames": 10,
                "sample_episodes": 100,
                "train_steps": 1000}

    writer = ExperimentWriter(agent_name="test_agent",
                              env_name="test_env",
                              exp_info="test_exp",
                              interval=interval)

    # GIVEN sample_frame_interval == 10
    # WHEN add_scalar with step="sample_frames" is called
    # THEN sample_frames is saved every 10 samples
    for i in range(100):
        writer.sample_frames = i
        writer.add_scalar("test", i, step="sample_frames")

    # same test for sample_episodes
    for i in range(1000):
        writer.sample_episodes = i
        writer.add_scalar("test", i, step="sample_episodes")

    # same test for train_steps
    for i in range(10000):
        writer.train_steps = i
        writer.add_scalar("test", i, step="train_steps")

    set_writer(writer)

    # load events file
    test_path = pathlib.Path("runs/test_exp")
    for p in test_path.rglob("events*"):
        eventspath = p
    event_acc = event_accumulator.EventAccumulator(
        str(eventspath), size_guidance={'scalars': 0})

    # test make dir
    assert os.path.isdir(str(test_path))

    yield event_acc

    # rm test_exp dir
    writer.close()
    rmtree(str(test_path), ignore_errors=True)


def read_scalars(event_acc):
    scalars = {}
    steps = {}

    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]
        steps[tag] = [event.step for event in events]
    return steps, scalars


def test_get_step_value(init_writer):
    writer = get_writer()
    writer.sample_frames = 1
    writer.sample_episodes = 2
    writer.train_steps = 3

    assert writer._get_step_value("sample_frames") == 1
    assert writer._get_step_value("sample_episodes") == 2
    assert writer._get_step_value("train_steps") == 3


def test_add_scalar_interval(init_writer):
    writer = get_writer()
    writer.close()

    event_acc = init_writer
    event_acc.Reload()

    steps, scalars = read_scalars(event_acc)
    assert scalars['test_env/test/sample_frames'] == [
        i for i in range(10, 100, 10)]

    assert scalars['test_env/test/sample_episodes'] == [
        i for i in range(100, 1000, 100)]

    assert scalars['test_env/test/train_steps'] == [
        i for i in range(1000, 10000, 1000)]


def test_step_value(init_writer):
    writer = get_writer()
    writer.sample_frames = 1e9
    writer.add_scalar("test", 999, step="sample_frames", step_value=12345)
    writer.close()

    event_acc = init_writer
    event_acc.Reload()
    steps, scalars = read_scalars(event_acc)
    assert 12345 in steps["test_env/test/sample_frames"]


def test_save_csv(init_writer):
    writer = get_writer()
    writer.sample_frames = 1e9
    writer.add_scalar("test", 500, step="sample_frames", save_csv=True)

    test_path = pathlib.Path("runs/test_exp")
    for p in test_path.rglob("*.csv"):
        csv_file = p

    csv_data = pd.read_csv(str(csv_file), names=["sample_frames", "return"])
    assert csv_data["sample_frames"].tolist() == [1e9]
