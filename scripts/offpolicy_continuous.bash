#!/bin/bash

export exp_info=off_policy
export train_minutes=60

for env in ant humanoid walker lander
    do
    for agent in ddpg td3 sac
        do
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 8 --exp_info $exp_info --device cuda:0 --seed 0 --trains_per_episode 5
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 8 --exp_info $exp_info --device cuda:0 --seed 1 --trains_per_episode 5
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 8 --exp_info $exp_info --device cuda:1 --seed 2 --trains_per_episode 5
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 8 --exp_info $exp_info --device cuda:1 --seed 3 --trains_per_episode 5
    done
done
