#!/bin/bash
for agent in ddpg td3 sac
    do 
    for num_trains_per_iter in 10 100 1000
        do
        tsp python ~/pytorch-rl-il/scripts/continuous.py ant $agent --frames 10000000 --num_workers 16 --exp_info evaluation --device cuda:0 --num_trains_per_iter $num_trains_per_iter --minibatch_size 1000
        tsp python ~/pytorch-rl-il/scripts/continuous.py ant $agent --frames 10000000 --num_workers 16 --exp_info evaluation --device cuda:1 --num_trains_per_iter $num_trains_per_iter --minibatch_size 100
    done
done