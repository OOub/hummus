from utilities import progress_bar, apply_rotations, print_in_logfile, AverageMeter, adjust_learning_rate
from log_reader import regression_spike_parser, regression_potential_parser
from random import shuffle
import multiprocessing
from time import time
from tqdm import tqdm
import numpy as np
import subprocess
import itertools
import os

# PyTorch dependency
from torch import nn
import torch
import torch.optim as optim
from torch.utils import data

#### parameters

# network type
feedforward = True
load_data = True
ds=40000

#### parsing data to be compatible with the logistic regression

if feedforward:
    naming = 'simple'
else:
    naming = 'deep'

if load_data:
    # path to log files
    training_filename = '/Users/omaroubari/Downloads/'+naming+'TrainingPLog.bin'
    test_filename = '/Users/omaroubari/Downloads/'+naming+'TestPLog.bin'
    
    # path to label files
    training_label_filename = '/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt'
    test_label_filename = '/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt'

    # import training data
    training_x = regression_potential_parser(training_filename, p=32, layer=1, N=100, ds=ds)
    training_labels = np.loadtxt(training_label_filename)
    training_labels[0,1] = 0

    print('finished importing training data')
    
    # parse training labels
    training_y = [None]*len(training_x)
    for i  in range(len(training_x)):
        for j in range(1, len(training_labels)+1):
            if j < len(training_labels):
                if i < training_labels[j,1] and i >= training_labels[j-1,1]:
                    training_y[i] = training_labels[j-1,0]
            else:
                if i < len(training_x) and i >= training_labels[j-1,1]:
                    training_y[i] = training_labels[j-1,0]

    print('finished parsing training data')

    # import test data
    test_x = regression_potential_parser(test_filename, p=32, layer=1, N=100, ds=ds)
    test_labels = np.loadtxt(test_label_filename)
    test_labels[0,1] = 0

    print('finished importing test data')
    
    # parse test labels
    test_y = [None]*len(test_x)
    for i  in range(len(test_x)):
        for j in range(1, len(test_labels)+1):
            if j < len(test_labels):
                if i < test_labels[j,1] and i >= test_labels[j-1,1]:
                    test_y[i] = test_labels[j-1,0]
                else:
                    if i < len(test_x) and i >= test_labels[j-1,1]:
                        test_y[i] = test_labels[j-1,0]

    print('finished parsing test data')

    # save files
    np.save(naming+'_training_x',training_x)
    np.save(naming+'_training_y',training_y)
    np.save(naming+'_test_x',test_x)
    np.save(naming+'_test_y',test_y)

    print('finished saving files training data')

else:
    training_x=np.load(naming+'_training_x'+'.npy').squeeze()
    training_y=np.load(naming+'_training_y'+'.npy').squeeze()
    test_x=np.load(naming+'_test_x'+'.npy').squeeze()
    test_y=np.load(naming+'_test_y'+'.npy').squeeze()

#### logistic regression
classifier = nn.Linear(training_x.shape[-1],10).double()
classifier_lr = 0.1
classifier_optimizer = optim.SGD(classifier.parameters(), lr=classifier_lr, momentum=0.9, weight_decay=5e-4)

classifier_best_acc = 0  # best test accuracy of the classifier on cifar
classifier_start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs_classifier=70

trainset = [(x,y) for x,y in zip(training_x,training_y)]
trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = [(x,y) for x,y in zip(test_x,test_y)]
testloader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
device='cpu'
criterion=nn.CrossEntropyLoss()
logfile = "logs_poker/log_"+naming+".log"
def train_classifier(epoch):
    global classifier_lr, classifier_optimizer
    print('\nEpoch: %d' % epoch)
    
    adjust_learning_rate(classifier_optimizer, epoch, classifier_lr, rate=0.1, adjust_frequency=50)
    
    classifier.train()
    train_losses = AverageMeter()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        classifier_optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # free stored activations for revnets
        
        classifier_optimizer.step()
        
        train_losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f (%.3f) | Acc: %.3f%% (%d/%d)'
                     % (train_losses.val, train_losses.avg, 100.*correct/total, correct, total))
    
    print_in_logfile('Epoch %d, Train, Loss: %.3f, Acc: %.3f' % (epoch, train_losses.avg , 100.*correct/total), logfile)


def test_classifier(epoch):
    global classifier_best_acc
    classifier.eval()
    test_losses = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = classifier(inputs)
            
            loss = criterion(outputs, targets)
            
            test_losses.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f (%.3f) | Acc: %.3f%% (%d/%d)'
                         % (test_losses.val, test_losses.avg, 100.*correct/total, correct, total))
        print_in_logfile('Epoch %d, Test,  Loss: %.3f, Acc: %.3f' % (epoch, test_losses.avg, 100.*correct/total), logfile)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > classifier_best_acc:
        print('Saving..')
        state = {
            'classifier': classifier.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/classifier_ckpt{}.t7'.format(endname))
        classifier_best_acc = acc

for epoch in range(100):
    if epoch==30:
        optimizer = optim.SGD(list(classifier.parameters()) , lr=classifier_lr/10.0, momentum=0.9,
                              weight_decay=5e-4)
    if epoch == 60:
        optimizer = optim.SGD(list(classifier.parameters()) , lr=classifier_lr/100.0, momentum=0.9,
                              weight_decay=5e-4)
    if epoch == 90:
        optimizer = optim.SGD(list(classifier.parameters()) , lr=classifier_lr/1000.0, momentum=0.9,
                              weight_decay=5e-4)
    
    train_classifier(epoch)
    test_classifier(epoch)
