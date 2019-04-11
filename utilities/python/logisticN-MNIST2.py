import sys
from tqdm import tqdm
from read_functions import read_spike_np2
import os
# import pandas as pd  # data handeling
import numpy as np   # numerical computing
# from scipy.optimize import minimize  # optimization code
# import matplotlib.pyplot as plt  # plotting
import itertools  # combinatorics functions for multinomial code
from random import shuffle
from utils import progress_bar
# Some model checking functions
#
from logistic_utils import  to_0_1, model_accuracy, model_accuracy_pos, model_accuracy_neg, \
    false_pos, false_neg, true_pos, true_neg, model_precision, \
        model_recall, print_model_quality, mean_normalize, apply_normalizer, gen_data

def running_mean(x, N):
    import ipdb;ipdb.set_trace()
    if len(x.shape)>1:
        cumsum = np.cumsum(np.insert(x, 0, np.zeros(x.shape[1]),axis=0),axis=0)  
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0))  
    return (cumsum[N:] - cumsum[:-N]) / float(N) 

#
# Main Logistic Regression Equations
#

N=1000
gpM=0
gpS=0.2
gM=0
gS=0.1
cF=2
cB=1
cS=1
I=100
P=120
R=0
w=1

#rcNetwork_N(100)_gp(0,0.2)_g(0,0.1)_c(2,1,1)_IPR(100,120,0)_w1.json
endname="_N({})_gp({},{})_g({},{})_c({},{},{})_IPR({},{},{})_w{}".format(N,gpM,gpS,gM,gS,cF,cB,cS,I,P,R,w)

ds=10000
endname+='_ds{}'.format(ds)

def one_hot_encode(labels_list, max_number):
    samples_number = len(labels_list)
    b = np.zeros((samples_number, max_number))
    b[np.arange(samples_number), labels_list] = 1
    return b

def softmax(x, epsilon=1e-9):
    e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return epsilon + e_x / e_x.sum(axis=1, keepdims=True)


def infer(W, X):
    ones = np.ones((len(X),1))
    Xplus1 = np.hstack((X, ones))
    Z = np.dot(Xplus1,W)
    Y = softmax(Z)
    return Y

def l1grad(w, fancy=False):
    gw=np.sign(w)
    if fancy:
        aw=np.abs(w)
        for l in range(10)[::-1]:
            gw[(aw>(4**l))&(aw<(4**(l+1)))]*=l+1
    return gw


def get_grad(W, X, Y, reg=None):
    h = infer(W, X)
    m = len(Y)
    eta = 1e-0
    ones = np.ones((len(X), 1))
    Xplus1 = np.hstack((X, ones))
    if reg=='l2':
        grad = (-1 / m) * np.dot(Xplus1.T, (Y - h)) + eta * W
    elif reg=='l1':
        grad = (-1 / m) * np.dot(Xplus1.T, (Y - h)) + eta * l1grad(W)
    else:
        grad = (-1 / m) * np.dot(Xplus1.T, (Y - h)) 
    return grad


X_train=np.load("X_train_pt{}.npy".format(endname)).squeeze()#[:,:30]
y_train=np.load("y_train_pt{}.npy".format(endname)).squeeze()
X_test=np.load("X_test_pt{}.npy".format(endname)).squeeze()#[:,:30]
y_test=np.load("y_test_pt{}.npy".format(endname)).squeeze()
# X_train=np.load("X_train{}.npy".format(endname)).squeeze()[:,:30]
# y_train=np.load("y_train{}.npy".format(endname)).squeeze()
# X_test=np.load("X_test{}.npy".format(endname)).squeeze()[:,:30]
# y_test=np.load("y_test{}.npy".format(endname)).squeeze()
# X_train=running_mean(X_train,100)
# y_train=np.round(running_mean(y_train.astype(np.float),100)).astype(np.int)
# Ntr=X_train.shape[0]
# X_test=running_mean(X_test,100)
# y_test=np.round(running_mean(y_test.astype(np.float),100)).astype(np.int)
n_classes = 10 #len(set(y_train))

SMALL=False
if SMALL:
    n_classes=2
    trflag=np.argwhere(y_train<n_classes).squeeze()
    teflag=np.argwhere(y_test<n_classes).squeeze()

    X_train=X_train[trflag]
    y_train=y_train[trflag]
    X_test=X_test[teflag]
    y_test=y_test[teflag]

    # ds = 1

    # Ntr=X_train.shape[0]//ds
    # Nte=X_test.shape[0]//ds
    # X_train = X_train[:Ntr*ds].reshape((-1,ds,X_train.shape[-1])).sum(1).squeeze()#[::ds_train]
    # y_train = y_train[:Ntr*ds][::ds]
    # X_test = X_test[:Nte*ds].reshape((-1,ds,X_test.shape[-1])).sum(1).squeeze()
    # y_test = y_test[:Nte*ds][::ds]

# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)

n_features = X_test.shape[1]
w = np.random.randn(n_features + 1, n_classes) / n_features
batch_size=1
num_epoch=100
step=1e-3
print("Number of datapoints {}".format(X_train.shape[0]))
print(endname)

from torch import nn
import torch
import torch.optim as optim
from torch.utils import data
from utils import progress_bar, apply_rotations, print_in_logfile, AverageMeter, adjust_learning_rate

classifier = nn.Linear(X_train.shape[-1],10).double()
classifier_lr = 0.1
classifier_optimizer = optim.SGD(classifier.parameters(), lr=classifier_lr, momentum=0.9, weight_decay=5e-4)

classifier_best_acc = 0  # best test accuracy of the classifier on cifar
classifier_start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs_classifier=70

trainset = [(x,y) for x,y in zip(X_train,y_train)]
trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


testset = [(x,y) for x,y in zip(X_test,y_test)]
testloader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
device='cpu'
criterion=nn.CrossEntropyLoss()
logfile = "logs/log_"+endname+".log"
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