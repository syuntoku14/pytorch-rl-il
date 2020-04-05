import unittest
from rlil.environments import GymEnvironment
from rlil.presets.batch_continuous import bcq
from rlil import nn
from rlil.environments import Action
from rlil.policies import DeterministicPolicy
from rlil.agents import GreedyAgent
import gym
from torch.optim import Adam


class TestContinuousPresets(unittest.TestCase):
    def test_bcq(self):
        env = gym.make('LunarLanderContinuous-v2')
        env = GymEnvironment(env)

        model = nn.Sequential(nn.Flatten(), nn.Linear(
            env.state_space.shape[0], Action.action_space().shape[0]))
        optimizer = Adam(model.parameters())
        greedy_agent = GreedyAgent(policy=DeterministicPolicy(
            model, optimizer, Action.action_space()))
        env.reset()

        # store samples
        while not env._state.done:
            action = greedy_agent.act(env.state, env.reward)
            state, reward = env.step(action)
        replay_buffer = greedy_agent.replay_buffer

        agent_fn = bcq(replay_buffer)
        agent = agent_fn(env)

        for _ in range(10):
            agent.train()


if __name__ == '__main__':
    unittest.main()
