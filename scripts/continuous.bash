#!/bin/bash
for env in humanoid
    do
    for agent in ddpg td3 sac
        do 
        tsp python ~/pytorch-rl-il/scripts/continuous.py $env $agent --frames 10000000 --num_workers 6 --exp_info ray_ddpg_td3_sac --device cuda:0 --num_trains_per_iter 10 --minibatch_size 1000
        tsp python ~/pytorch-rl-il/scripts/continuous.py $env $agent --frames 10000000 --num_workers 6 --exp_info ray_ddpg_td3_sac --device cuda:1 --num_trains_per_iter 10 --minibatch_size 1000
    done
done