#!/bin/bash

export exp_info=offline
export train_minutes=120

for agent in bc bcq
    do
    for seed in {0..3}
        do
        # ant
        tsp python ~/pytorch-rl-il/scripts/continuous/offline.py ant $agent runs/demos/AntBulletEnv-v0/td3_3000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
        # hopper
        tsp python ~/pytorch-rl-il/scripts/continuous/offline.py hopper $agent runs/demos/HopperBulletEnv-v0/sac_2000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
        # humanoid
        tsp python ~/pytorch-rl-il/scripts/continuous/offline.py humanoid $agent runs/demos/HumanoidBulletEnv-v0/td3_1700 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
        # walker
        tsp python ~/pytorch-rl-il/scripts/continuous/offline.py walker $agent runs/demos/WalkerBulletEnv-v0/ppo_3000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
    done
done