import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import csv
import os
import shutil
import random
import argparse
from models import *
from load_datasets import *

parser = argparse.ArgumentParser(description='CIFAR training')
parser.add_argument('exp_name', type=str, help='Experiment name')
parser.add_argument('--model', type=str, default='resnet', choices=( 'vgg', 'resnet'), help='Which network model to use')
parser.add_argument('--validate-interval', type=int, default=9999, help='number of iterations between validation')
parser.add_argument('--dump-interval', default=50, type=int, help='epochs between consecutive dumping')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1234)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing (default: 500)')
parser.add_argument('--val-batch-size', type=int, default=128, help='input batch size for validation (default: 500)')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--resume-training', action='store_true', default=False, help='Resume training from file (if present)')
parser.add_argument('--lr-schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
parser.add_argument('--lr-drops', type=float, nargs='+', default=[], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--save-network', action='store_true', default=False, help='Save network')
parser.add_argument('--dump', action='store_true', default=False, help='dump results in csv files')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay.')
parser.add_argument('--val-set-size', type=int, default=5000, help='how much of the training set to use for validation (default: 5000)')
parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train (default: 40)')
parser.add_argument('--pr-ft-epochs', type=int, default=40, help='number of fine-tuning epochs when pruning (default: 40)')
parser.add_argument('--dump-pr', action='store_true', default=False, help='dump acc after pruning (before rewinding)')
parser.add_argument('--pruned-path', type=str, help='path to pruned file')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

batch_size = 128 if args.model == 'resnet' else 64

path = 'Logs/{0}/{1}'.format(args.exp_name, args.seed)
if args.resume_training and os.path.isfile(os.path.join(path,'end.txt')):
    exit()

train_loader, val_loader, test_loader = generate_loaders('cifar10', args.val_set_size, batch_size, args.val_batch_size, args.test_batch_size, args.workers, augment=(args.model=='resnet' or args.model=='vgg'))

t_path = 'Logs/{0}/{1}'.format(args.exp_name, args.seed)
if args.resume_training and os.path.isfile(os.path.join(t_path,'end.txt')):
    exit()

model_dict = {'resnet': ResNet, 'vgg': VGG}

model = model_dict[args.model]()
if args.cuda: model.cuda()

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    assert len(args.lr_schedule) == len(args.lr_drops), "length of gammas and schedule should be equal"
    for (drop, step) in zip(args.lr_drops, args.lr_schedule):
        if (epoch >= step): lr = lr * drop
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr

def group_weight_decay(model, skip_list):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if sum([pattern in name for pattern in skip_list]) > 0:
            no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': args.decay}]

def compute_remaining_weights():

    zeros = 0.
    numel = 0.

    for m in model.prunable:
        zeros += (m.weight.data == 0).float().sum()
        numel += np.prod(m.weight.data.size())         

    return 1 - zeros/numel  

def finish_run():
    if args.dump: writeFile.close()
    if args.save_network:
        # os.system("rm {}".format(os.path.join(path,'training_state.tar')))
        # os.system("rm {}".format(os.path.join(path,'training_state_temp.tar')))
        os.system("touch {}".format(os.path.join(path,'end.txt')))

def train(checkpoint, prune_fine_tuning_phase = False, masks = None):
    iterations_counter = 0
    remaining_weights = 1.
    if prune_fine_tuning_phase:
        epoch = args.epochs
        max_epochs = args.epochs + args.pr_ft_epochs
    else:
        epoch = 0
        max_epochs = args.epochs

    if args.dump: rows_to_write = []
    while epoch < max_epochs:

        if checkpoint:
            if epoch == 0 and args.resume_training:
                epoch = checkpoint['epoch'] + 1
                iterations_counter = checkpoint['iterations_counter']
                remaining_weights = checkpoint['remaining_weights']
                if epoch == max_epochs:
                    break
        print('\t--------- Epoch {} -----------'.format(epoch))
        
        adjust_learning_rate(optimizer, epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            if args.cuda: data, target = data.cuda(), target.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1)[1]

            batch_correct = pred.eq(target.data.view_as(pred)).sum()

            loss = F.cross_entropy(output, target)
            loss.backward()

            if prune_fine_tuning_phase:
                for m_i, m in enumerate(model.prunable):
                    m.weight.grad = m.weight.grad * masks[m_i]

            optimizer.step()

            remaining_weights = compute_remaining_weights()
            if (iterations_counter > 0 and iterations_counter % args.validate_interval == 0) or batch_idx == len(train_loader) - 1:
                val_loss, val_acc, _ = test(val_loader)
                test_loss, test_acc, _ = test(test_loader)
                
                print('\t\tRemaining weights: {:.4f}\tVal acc: {:.1f}\tTest acc: {}'.format(remaining_weights, val_acc, test_acc))
                
                if args.dump:
                    rows_to_write.append([epoch, iterations_counter, remaining_weights, val_acc, val_loss, test_acc, test_loss])
                
            iterations_counter += 1

        if args.dump and (epoch % args.dump_interval == 0 or epoch == max_epochs -1):
            for row in rows_to_write: writer.writerow(row)
            writeFile.flush()
            os.fsync(writeFile.fileno()) 
            rows_to_write = []            

        if args.save_network and (epoch % args.dump_interval == 0 or epoch == max_epochs -1):
  
            torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'iterations_counter': iterations_counter,
            'remaining_weights': remaining_weights,
            'opt_state_dict': optimizer.state_dict(),
            'prune_fine_tuning_phase': prune_fine_tuning_phase
            }, os.path.join(path,'training_state_temp.tar'))

            os.rename(os.path.join(path,'training_state_temp.tar'), os.path.join(path,'training_state.tar'))

        epoch += 1

def test(loader):
    model.eval()
    loss = 0.
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data, target in loader:
            if args.cuda: data, target = data.cuda(), target.cuda(non_blocking=True)
            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False)
            pred = output.max(1)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += data.size()[0]

    remaining_weights = compute_remaining_weights()
    
    loss /= total
    acc = 100. * correct.item() / total

    return loss.item(), acc, remaining_weights

if args.dump:
    path = 'Logs/{0}/{1}'.format(args.exp_name, args.seed)
    if os.path.isdir(path) and args.resume_training == False:
        shutil.rmtree(path)        
    if os.path.isdir(path) == False:
        os.makedirs(path)

    command_file = open(os.path.join(path,'command_run.txt'), 'w')
    command_file.write(str(args))
    command_file.close()        

    if args.resume_training == False:
        writeFile = open(os.path.join(path,'cifar.csv'), 'w')
        writer = csv.writer(writeFile)
        writer.writerow(['Epoch','Iteration','WeightsRemaining','ValAcc','ValLoss','TestAcc','TestLoss'])
        if args.dump_pr:
            pr_writeFile = open(os.path.join(path,'pr.csv'), 'w')
            pr_writer = csv.writer(pr_writeFile)
            pr_writer.writerow(['WeightsRemaining','ValAcc','ValLoss','TestAcc','TestLoss'])
    else:        
        log_present = os.path.isfile( os.path.join(path,'cifar.csv') )
        writeFile = open(os.path.join(path,'cifar.csv'), 'a')
        writer = csv.writer(writeFile)
        if log_present == False:
            writer.writerow(['Epoch','Iteration','WeightsRemaining','ValAcc','ValLoss','TestAcc','TestLoss'])
        if args.dump_pr:
            pr_log_present = os.path.isfile( os.path.join(path,'pr.csv') )
            pr_writeFile = open(os.path.join(path,'pr.csv'), 'a')
            pr_writer = csv.writer(pr_writeFile)
            if pr_log_present == False:
                pr_writer.writerow(['WeightsRemaining','ValAcc','ValLoss','TestAcc','TestLoss'])            

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

num_params = sum([p.numel() for p in trainable_params])
print("Total number of parameters: {}".format(num_params))

checkpoint = None
prune_fine_tuning_phase = False

optimizer = optim.SGD(trainable_params, lr=args.lr, weight_decay=args.decay)

if args.resume_training and os.path.isfile(os.path.join(path,'training_state.tar') ):
    checkpoint = torch.load(os.path.join(path,'training_state.tar'))
    prune_fine_tuning_phase = checkpoint['prune_fine_tuning_phase']
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])

def dump_prune_routine():

    save_lr_sch = args.lr_schedule
    for i in range(len(args.lr_schedule)):
            args.lr_schedule[i] = -1

    assert os.path.isfile(args.pruned_path)

    checkpoint = torch.load(args.pruned_path)

    model.load_state_dict(checkpoint) 

    masks = []
    for m in model.prunable:
        masks.append( (m.weight.data != 0.).float() )

      
    train(checkpoint, prune_fine_tuning_phase = True, masks = masks)        

    val_loss, val_acc, _ = test(val_loader)
    test_loss, test_acc, remaining_weights = test(test_loader)
    args.lr_schedule = save_lr_sch
    pr_writer.writerow([remaining_weights, val_acc, val_loss, test_acc, test_loss])
    pr_writeFile.flush()
    os.fsync(pr_writeFile.fileno())   
    finish_run()

if not prune_fine_tuning_phase and not args.dump_pr:
    train(checkpoint, prune_fine_tuning_phase = False)

if args.dump_pr:
    dump_prune_routine()
else:
    finish_run()