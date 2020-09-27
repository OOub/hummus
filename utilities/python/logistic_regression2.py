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

    def __init__(self, n_in, n_out, learning_rate = 0.1, weight_decay=0.01, batch_size=128, epochs=70):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = LogisticRegression(n_in, n_out)
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, x, y):
        tensor_x= torch.from_numpy(x)
        tensor_y= torch.from_numpy(y)

        my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
        my_dataloader = data.DataLoader(my_dataset, batch_size=self.batch_size) # create your dataloader

        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            for features, labels in my_dataloader:

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

trials = 5
dpts = [1000,5000]
base_path = "/home/omaroubari/repositories/hummus/"
base_name = "conductance_nmnist_75_5000dp"

if trials == 1:
    for k in dpts:
        acc = []
        trd = np.load(base_path+base_name+"_tr_set.npy").astype(np.float32)
        trl = np.load(base_path+base_name+"_tr_label.npy").astype(np.int64)
        ted = np.load(base_path+base_name+"_te_set.npy").astype(np.float32)
        tel = np.load(base_path+base_name+"_te_label.npy").astype(np.int64)

        lreg = LogReg(n_in=trd.shape[1],n_out=np.unique(trl).shape[0])
        lreg.fit(trd[-k:,:],trl[-k:])
        acc.append(((lreg.predict(ted).numpy()==tel).sum()/tel.shape[0])*100)
        print(acc,"for %s datapoints" % k)
elif trials > 1: 
    for k in dpts:
        acc = []
        for i in range(trials):
            trd = np.load(base_path+base_name+str(i)+"_tr_set.npy").astype(np.float32)
            trl = np.load(base_path+base_name+str(i)+"_tr_label.npy").astype(np.int64)
            ted = np.load(base_path+base_name+str(i)+"_te_set.npy").astype(np.float32)
            tel = np.load(base_path+base_name+str(i)+"_te_label.npy").astype(np.int64)

            lreg = LogReg(n_in=trd.shape[1],n_out=np.unique(trl).shape[0])
            lreg.fit(trd[-k:,:],trl[-k:])
            acc.append(((lreg.predict(ted).numpy()==tel).sum()/tel.shape[0])*100)

        acc = np.array(acc)
        print(np.mean(acc),'\u00B1',np.std(acc),"for %s datapoints" % k)
else:
    print("wrong number of trials")