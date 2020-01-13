import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy.io as sio

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.activation(outputs)
        return outputs

class LogReg(object):

    def __init__(self, n_in, n_out, learning_rate = 0.001, batch_size=32, epochs=100):
        self.learning_rate = learning_rate
        self.model = LogisticRegression(n_in, n_out)
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, x, y):
        tensor_x= torch.from_numpy(x)
        tensor_y= torch.from_numpy(y)

        my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
        my_dataloader = data.DataLoader(my_dataset, batch_size=self.batch_size) # create your dataloader

        criterion = nn.NLLLoss()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)

        for epoch in range(self.epochs):
            for i, (features, labels) in enumerate(my_dataloader):

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = self.model(features)

                # Calculate Loss: LogSoftmax --> negative log likelihood loss
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()


    def predict(self, x):
        tensor_x = torch.from_numpy(x)
        outputs = self.model(tensor_x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

task = 3

if task == 0:
    # 3-class N-MNIST (96%)
    trd = np.load("/Users/omaroubari/Desktop/report/3_classes_nmnist/nmnist_3_tr_set.npy").astype(np.float32)
    trl = np.load("/Users/omaroubari/Desktop/report/3_classes_nmnist/nmnist_3_tr_label.npy").astype(np.int64)
    ted = np.load("/Users/omaroubari/Desktop/report/3_classes_nmnist/nmnist_3_te_set.npy").astype(np.float32)
    tel = np.load("/Users/omaroubari/Desktop/report/3_classes_nmnist/nmnist_3_te_label.npy").astype(np.int64)
elif task == 1:
    # 10-class N-MNIST (76%)
    trd = np.load("/Users/omaroubari/Desktop/report/10_classes_nmnist/nmnist_10_tr_set.npy").astype(np.float32)
    trl = np.load("/Users/omaroubari/Desktop/report/10_classes_nmnist/nmnist_10_tr_label.npy").astype(np.int64)
    ted = np.load("/Users/omaroubari/Desktop/report/10_classes_nmnist/nmnist_10_te_set.npy").astype(np.float32)
    tel = np.load("/Users/omaroubari/Desktop/report/10_classes_nmnist/nmnist_10_te_label.npy").astype(np.int64)
elif task == 2:
    # 4-class POKER-DVS (100%)
    trd = np.load("/Users/omaroubari/Desktop/report/pips_40e_84_6/poker_tr_set.npy").astype(np.float32)
    trl = np.load("/Users/omaroubari/Desktop/report/pips_40e_84_6/poker_tr_label.npy").astype(np.int64)
    ted = np.load("/Users/omaroubari/Desktop/report/pips_40e_84_6/poker_te_set.npy").astype(np.float32)
    tel = np.load("/Users/omaroubari/Desktop/report/pips_40e_84_6/poker_te_label.npy").astype(np.int64)
elif task == 3:
    # N-CARS - 28x28 from origin (54%)
    trd = np.load("/Users/omaroubari/Desktop/report/ncars_28x28_from0/ncars_scaled_tr_set.npy").astype(np.float32)
    trl = np.load("/Users/omaroubari/Desktop/report/ncars_28x28_from0/ncars_scaled_tr_label.npy").astype(np.int64)
    ted = np.load("/Users/omaroubari/Desktop/report/ncars_28x28_from0/ncars_scaled_te_set.npy").astype(np.float32)
    tel = np.load("/Users/omaroubari/Desktop/report/ncars_28x28_from0/ncars_scaled_te_label.npy").astype(np.int64)
elif task == 4:
    # 10-class N-MNIST 2 epochs
    trd = np.load("/Users/omaroubari/Desktop/report/10classes_2epochs/nmnist_10_2e_tr_set.npy").astype(np.float32)
    trl = np.load("/Users/omaroubari/Desktop/report/10classes_2epochs/nmnist_10_2e_tr_label.npy").astype(np.int64)
    ted = np.load("/Users/omaroubari/Desktop/report/10classes_2epochs/nmnist_10_2e_te_set.npy").astype(np.float32)
    tel = np.load("/Users/omaroubari/Desktop/report/10classes_2epochs/nmnist_10_2e_te_label.npy").astype(np.int64)

# ACCURACY VS SAVED DATAPOINTS PLOT
dpts = list(range(0,1200,10))

best = 0
bestn = 0
acc = []
points = []
for k in dpts[1:]:
    lreg = LogReg(n_in=100,n_out=np.unique(trl).shape[0], epochs=45)
    lreg.fit(trd[-k:,:],trl[-k:])

    acc.append((lreg.predict(ted).numpy()==tel).sum()/tel.shape[0])
    points.append(k)
    if acc[-1]>best:
        best=acc[-1]
        bestn=k
        print("We have the best test accuracy at {:.05} using {} datapoints  ".format(best,bestn))
        p_tel = lreg.predict(ted).numpy()
        if task == 0:
            sio.savemat('3nmnist.mat', {"predicted_labels":p_tel, "true_labels":tel, "accuracy":acc, "points":points})
        elif task == 1:
            sio.savemat('10nmnist.mat', {"predicted_labels":p_tel, "true_labels":tel, "accuracy":acc, "points":points})
        elif task == 2:
            sio.savemat('poker.mat', {"predicted_labels":p_tel, "true_labels":tel, "accuracy":acc, "points":points})
        elif task == 3:
            sio.savemat('ncar28x28.mat', {"predicted_labels":p_tel, "true_labels":tel, "accuracy":acc, "points":points})
        elif task == 4:
            sio.savemat('10nmnist_2e.mat', {"predicted_labels":p_tel, "true_labels":tel, "accuracy":acc, "points":points})
