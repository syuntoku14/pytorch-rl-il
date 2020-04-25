#!/bin/bash

export exp_info=cpprb
export num_trains=5

for env in ant humanoid walker lander
    do
    for agent in ddpg td3 sac
        do
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_frames 50000000 --num_workers 8 --exp_info $exp_info --device cuda:0 --num_trains_per_episode $num_trains --minibatch_size 500 --seed 0
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_frames 50000000 --num_workers 8 --exp_info $exp_info --device cuda:0 --num_trains_per_episode $num_trains --minibatch_size 500 --seed 1
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_frames 50000000 --num_workers 8 --exp_info $exp_info --device cuda:1 --num_trains_per_episode $num_trains --minibatch_size 500 --seed 2
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_frames 50000000 --num_workers 8 --exp_info $exp_info --device cuda:1 --num_trains_per_episode $num_trains --minibatch_size 500 --seed 3
    done
done