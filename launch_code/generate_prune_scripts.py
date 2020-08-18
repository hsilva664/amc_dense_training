import os

models = ['vgg', 'resnet']
seeds = [1234, 2234, 3234]
wrem_list = [0.5, 0.25, 0.14, 0.07, 0.03, 0.02, 0.01, 0.005]

# python train_cifar.py {exp_name} --model {model} --seed {seed} --dump --dump-interval 10 --lr-schedule 80 120 --lr-drops 0.1 0.1 --dump-pr --pruned-path {pruned_path}

for seed in seeds:
    for wrem in wrem_list:
        for model in models:
            exp_name = model + '_wrem_' + str(wrem)
            pruned_path = os.path.join('Logs',model,str(seed),'pruned_{}.tar'.format(wrem))
            print 'python train_cifar.py {exp_name} --model {model} --seed {seed} --dump --dump-interval 10 --lr-schedule 80 120 --lr-drops 0.1 0.1 --dump-pr --pruned-path {pruned_path}'.format(model=model, seed=seed, exp_name=exp_name, pruned_path=pruned_path)