#!/bin/bash

export exp_info=on_policy
export train_minutes=60

for env in ant humanoid walker lander
    do
    for agent in ppo
        do
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 5 --exp_info $exp_info --device cuda:0 --seed 1
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 5 --exp_info $exp_info --device cuda:0 --seed 2
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 5 --exp_info $exp_info --device cuda:0 --seed 3
        tsp python ~/pytorch-rl-il/scripts/training/online_continuous.py $env $agent --train_minutes $train_minutes --num_workers 5 --exp_info $exp_info --device cuda:0 --seed 4
    done
done
