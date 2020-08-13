import os

models = ['vgg', 'resnet']
seeds = [1234, 2234, 3234]

# VARIABLE ARGS: exp_name --model --seed
# FIXED ARGS: --resume-training --save-network --lr-schedule 80 120 --lr-drops 0.1 0.1  --dump
# DO NOT USE: --dump-pr

for model in models:
    for seed in seeds:
        exp_name = 'dense_' + model
        print 'python train_cifar.py {exp_name} --model {model} --seed {seed} --resume-training --save-network --lr-schedule 80 120 --lr-drops 0.1 0.1 --dump'.format(model=model, seed=seed, exp_name=exp_name)