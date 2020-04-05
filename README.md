# TODO: Better documentation
# TODO: Replace unittest with pytest

# PyTorch-RL-IL (rlil): A PyTorch Library for Building Reinforcement Learning and Imitation Learning Agents

`rlil` is a library for reinforcement learning and imitation learning research. 
**Most of the codes of this library are the same as the [Autonomous Learning Library (ALL)](https://github.com/cpnota/autonomous-learning-library/tree/master/all)**
See the original documentation for the basic concepts of the library.


## Algorithms

### Reinforcement Learning

- [x] `Twind Dueling DDPG (TD3)`: https://arxiv.org/abs/1802.09477

### Imitation Learning

- [x] `Behavioral Cloning (BC)`
- [ ] `Generative Adversarial Imitation Learning (GAIL)` : https://arxiv.org/abs/1606.03476

### Others

- [ ] `Clipped Action Policy Gradient (CAPG)`: https://arxiv.org/abs/1802.07564

### New features of RLIL

- `ParallelEnvRunner` collects samples with multi-process environments. 
- `Action` class for better action handling

## Imitation Learning

### Collect trajectory

To collect the trajectory during watch_continuous.py, pass the option `--save_buffer` as:
```
python scripts/watch_continuous.py env dir --save_buffer
```

### Behavioral Cloning

`behavioral_cloning.py` trains the policy network using the collected replay_buffer.

```
python scripts/behavioral_cloning.py env agent dir
```

### Use pre-trained policy

To use the pre-trained policy state_dict, pass the `--policy` option to continuous.py as:

```
python scripts/continuous.py env agent --policy [path to the policy state_dict]
```