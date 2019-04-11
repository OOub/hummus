# import pandas as pd  # data handeling
import numpy as np   # numerical computing
# from scipy.optimize import minimize  # optimization code
# import matplotlib.pyplot as plt  # plotting
# import itertools  # combinatorics functions for multinomial code

# Some model checking functions
#
def to_0_1(h_prob) : # convert probabilites to true (1) or false (0) at cut-off 0.5
    return np.where(h_prob >= 0.5, 1, 0)

def model_accuracy(h,y) : # Overall accuracy of model
    return np.sum(h==y)/y.size * 100

def model_accuracy_pos(h,y) : # Accuracy on positive cases
    return np.sum(y[h==1] == 1)/y[y==1].size * 100

def model_accuracy_neg(h,y) : # Accuracy on negative cases
    return np.sum(y[h==0] == 0)/y[y==0].size * 100

def false_pos(h,y) : # Number of false positives
    return np.sum((y==0) & (h==1))

def false_neg(h,y) : # Number of false negatives
    return np.sum((y==1) & (h==0))

def true_pos(h,y) : # Number of true positives
    return np.sum((y==1) & (h==1))

def true_neg(h,y) : # Number of true negatives
    return np.sum((y==0) & (h==0))

def model_precision(h,y) : # Precision = TP/(TP+FP)
    return true_pos(h,y)/(true_pos(h,y) + false_pos(h,y))

def model_recall(h,y) : # Recall = TP/(TP+FN)
    return true_pos(h,y)/(true_pos(h,y) + false_neg(h,y))

def print_model_quality(title, h, y) : # Print the results of the functions above
    print( '\n# \n# {} \n#'.format(title) )
    print( 'Total number of data points   = {}'.format(y.size))
    print( 'Number of Positive values(1s) = {}'.format(y[y==1].size))
    print( 'Number of Negative values(0s) = {}'.format(y[y==0].size))
    print( '\nNumber of True Positives = {}'.format(true_pos(h,y)) )
    print( 'Number of False Positives = {}'.format(false_pos(h,y)) )
    print( '\nNumber of True Negatives = {}'.format(true_neg(h,y)) )
    print( 'Number of False Negatives = {}'.format(false_neg(h,y)) )
    print( '\nModel Accuracy = {:.2f}%'.format( model_accuracy(h,y) ) )
    print( 'Model Accuracy Positive Cases = {:.2f}%'.format( model_accuracy_pos(h,y) ) )
    print( 'Model Accuracy Negative Cases = {:.2f}%'.format( model_accuracy_neg(h,y) ) )
    print( '\nModel Precision = {}'.format(model_precision(h,y)) )
    print( '\nModel Recall = {}'.format(model_recall(h,y)) )


def mean_normalize(X):
    '''apply mean normalization to each column of the matrix X'''
    X_mean=X.mean(axis=0)
    X_std=X.std(axis=0)
    return (X-X_mean)/X_std

def apply_normalizer(X,X_mean,X_std) :
    return (X-X_mean)/X_std

def gen_data(N=1000,D=2,C=2):
    labels=np.zeros((N,C))
    W=np.random.randn(C,D)
    _, W= np.linalg.qr(W,mode='complete')
    # W=np.linalg.inv(np.dot(W.T)
    for i in range(N):
        labels[i,np.random.randint(C)]=1
    features=np.dot(labels,W)+np.random.randn(N,D)*0.1
    return features,labels

    