from rlil.environments import GymEnvironment, ENVS
from rlil.presets import continuous
from rlil.utils.optim import EarlyStopping
from rlil.initializer import get_logger
from rlil import nn
import torch
import pickle
import argparse
import os
import logging
import pybullet
import pybullet_envs
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split


# TODO: Use config file for logging
# TODO: Merge to batch_rl presets

class BCDataset(Dataset):
    def __init__(self, replay_buffer_path):
        with open(replay_buffer_path, mode="rb") as f:
            replay_buffer = pickle.load(f)
            expert_buffer = replay_buffer
        self.inputs = [item[0].features for item in expert_buffer]
        self.targets = [item[1].features for item in expert_buffer]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        state = self.inputs[idx].squeeze(0)
        action = self.targets[idx].squeeze(0)
        return state, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Name of the env (see envs)")
    parser.add_argument(
        "agent",
        help="Name of the agent (e.g. actor_critic). See presets for available agents.",
    )
    parser.add_argument(
        "dir", help="Directory where the agent's model was saved.")
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--train_iters", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # set logger
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.dir, "BC.log"))
    fmt = logging.Formatter('%(levelname)s : %(asctime)s : %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # initialize policy
    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id)
    agent_name = args.agent
    preset = getattr(continuous, agent_name)
    agent_fn = preset(device=args.device)
    agent = agent_fn(env)
    net = agent.policy.model.model.to(args.device)

    # set dataset
    dataset = BCDataset(os.path.join(args.dir, "buffer.pkl"))
    val_ratio = 0.1
    train_set, val_set = random_split(dataset,
                                      [int(len(dataset) * (1-val_ratio)), int(len(dataset) * val_ratio)])
    train_loader = DataLoader(
        train_set, batch_size=args.minibatch_size, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=args.minibatch_size, shuffle=True)

    # training iteration
    criterion = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    early_stopping = EarlyStopping(verbose=True,
                                   file_name=os.path.join(args.dir, "BC_state_dict.pt"))

    for epoch in tqdm(range(args.train_iters)):
        train_loss = 0.0
        val_loss = 0.0

        # training
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)[:, :].squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

        # validation
        for j, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # forward
            outputs = net(inputs)[:, :].squeeze()
            loss = criterion(outputs, labels)

            # print statistics
            val_loss += loss.item()

        logger.info('[%d, %5d] train loss: %.3f' %
                    (epoch + 1, i + 1, train_loss / i))
        logger.info('[%d, %5d] val loss: %.3f' %
                    (epoch + 1, j + 1, val_loss / j))
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    logger.info('training done')


if __name__ == "__main__":
    main()
