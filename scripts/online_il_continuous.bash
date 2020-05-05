#!/bin/bash

export exp_info=online_il
export train_minutes=120
export trains_per_episode=5
export num_workers=8

for agent in gail sqil 
    do
    for base_agent in ppo sac
        do
        for seed in {0..3}
            do
            # ant
            tsp python ~/pytorch-rl-il/scripts/continuous/online_il.py ant $agent $base_agent runs/demos/AntBulletEnv-v0/td3_3000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
            # hopper
            tsp python ~/pytorch-rl-il/scripts/continuous/online_il.py hopper $agent $base_agent runs/demos/HopperBulletEnv-v0/sac_2000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:0 --seed $seed 
            # humanoid
            tsp python ~/pytorch-rl-il/scripts/continuous/online_il.py humanoid $agent $base_agent runs/demos/HumanoidBulletEnv-v0/td3_1700 --train_minutes $train_minutes --exp_info $exp_info --device cuda:1 --seed $seed 
            # walker
            tsp python ~/pytorch-rl-il/scripts/continuous/online_il.py walker $agent $base_agent runs/demos/Walker2DBulletEnv-v0/ppo_3000 --train_minutes $train_minutes --exp_info $exp_info --device cuda:1 --seed $seed 
        done
    done
done