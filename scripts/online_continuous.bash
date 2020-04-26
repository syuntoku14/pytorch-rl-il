#!/bin/bash

export exp_info=new_train_per_sample

for env in ant humanoid walker lander
    do
    for agent in ddpg td3 sac
        do
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_steps 10000000 --num_workers 8 --exp_info $exp_info --device cuda:0 --seed 0
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_steps 10000000 --num_workers 8 --exp_info $exp_info --device cuda:0 --seed 1
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_steps 10000000 --num_workers 8 --exp_info $exp_info --device cuda:1 --seed 2
        tsp python ~/pytorch-rl-il/scripts/online_continuous.py $env $agent --train_steps 10000000 --num_workers 8 --exp_info $exp_info --device cuda:1 --seed 3
    done
done