#!/bin/bash

export exp_info=cpprb

for env in ant humanoid
    do
    for agent in ddpg td3 sac
        do 
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --frames 10000000 --num_workers 8 --exp_info $exp_info --device cuda:0 --num_trains_per_iter 10 --minibatch_size 1000
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --frames 10000000 --num_workers 8 --exp_info $exp_info --device cuda:1 --num_trains_per_iter 10 --minibatch_size 1000
    done
done