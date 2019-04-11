
import pandas as pd  # data handeling
import numpy as np   # numerical computing
from scipy.optimize import minimize  # optimization code
import matplotlib.pyplot as plt  # plotting
import itertools  # combinatorics functions for multinomial code

#
# Some model checking functions
#
from logistic_utils import  to_0_1, model_accuracy, model_accuracy_pos, model_accuracy_neg, \
    false_pos, false_neg, true_pos, true_neg, model_precision, \
        model_recall, print_model_quality, mean_normalize, apply_normalizer, gen_data

#
# Main Logistic Regression Equations
#


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

trdata = pd.read_csv("./fashion_train.csv")
trdata.head()
tedata = pd.read_csv("./fashion_test.csv")
tedata.head()
# data = data[data.label.isin([5,6,7,8])]
# data.label -=5
X_train, y_train = np.array(trdata.iloc[:, :-1]), np.array(trdata.iloc[:,-1])
X_test, y_test = np.array(tedata.iloc[:, :-1]), np.array(tedata.iloc[:,-1])

X_mean=np.mean(X_train,0,keepdims=True)
X_std=np.std(X_train,0,keepdims=True)+1
# X_train = (X_train -X_mean)#/X_std
# X_test = (X_test -X_mean)#/X_std

# X_train, X_test, y_train, y_test = train_test_split(X, Y)
n_classes = len(set(y_train))
n_features = X_test.shape[1]

w = np.random.randn(n_features + 1, n_classes) / n_features

batch_size=1
num_epoch=3000
step=1e-3

import tqdm
for epoch in tqdm.tqdm(range(num_epoch)):
    for iter_num, (x_batch, y_batch) in enumerate(zip(np.split(X_train, batch_size), np.split(y_train, batch_size))):
        y_batch_onehot = one_hot_encode(y_batch, n_classes)
        grad_batch = get_grad(w, x_batch, y_batch_onehot,reg='l1')
        w = w - step * grad_batch
        # w[w<1e-5]=0.0

y_pred1h=np.round(infer(w,X_test))
y_pred = np.argwhere(y_pred1h)[:,1]
print("Accuracy: {}".format(float(np.sum(y_pred==y_test))/y_pred.size ))

from sklearn.linear_model import LogisticRegression

lreg=LogisticRegression(C=0.1,penalty='l1')
lreg.fit(X_train,y_train)
y_pred2=lreg.predict(X_test)
print("Accuracy sk: {}".format(float(np.sum(y_pred2==y_test))/y_pred2.size ))
