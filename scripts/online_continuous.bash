#!/bin/bash

export exp_info=online
export train_minutes=120
export num_workers=8

for env in ant humanoid walker lander
    do
    for agent in ppo ddpg td3 sac
        do
        tsp python ~/pytorch-rl-il/scripts/continuous/online.py $env $agent --train_minutes $train_minutes --num_workers $num_workers --exp_info $exp_info --device cuda:0 --seed 0
        tsp python ~/pytorch-rl-il/scripts/continuous/online.py $env $agent --train_minutes $train_minutes --num_workers $num_workers --exp_info $exp_info --device cuda:0 --seed 1
        tsp python ~/pytorch-rl-il/scripts/continuous/online.py $env $agent --train_minutes $train_minutes --num_workers $num_workers --exp_info $exp_info --device cuda:1 --seed 2
        tsp python ~/pytorch-rl-il/scripts/continuous/online.py $env $agent --train_minutes $train_minutes --num_workers $num_workers --exp_info $exp_info --device cuda:1 --seed 3
    done
done
