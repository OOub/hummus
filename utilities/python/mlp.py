import torch
import torch.nn as nn
import numpy as np
from torch.utils import data

class MLP2(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(MLP2, self).__init__()
		H=50
		self.linear1 = torch.nn.Linear(input_dim, H)
		# self.activation1 = torch.nn.Sigmoid()
		# self.linear1 = torch.nn.Linear(input_dim, 20)
		self.activation1 = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(H, output_dim)
		self.activation2 = nn.LogSoftmax(dim=1)

	def forward(self, x):
		outputs = self.linear1(x)
		outputs = self.activation1(outputs)
		outputs = self.linear2(outputs)
		outputs = self.activation2(outputs)
		return outputs

class MLP3(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(MLP3, self).__init__()
		H=50
		self.linear1 = torch.nn.Linear(input_dim, H)
		# self.activation1 = torch.nn.Sigmoid()
		# self.linear1 = torch.nn.Linear(input_dim, 20)
		self.activation1 = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(H, H)
		self.activation2 = nn.ReLU()
		self.linear3 = torch.nn.Linear(H, output_dim)
		self.activation3 = nn.LogSoftmax(dim=1)

	def forward(self, x):
		outputs = self.linear1(x)
		outputs = self.activation1(outputs)
		outputs = self.linear2(outputs)
		outputs = self.activation2(outputs)
		outputs = self.linear3(outputs)
		outputs = self.activation3(outputs)
		return outputs

class MLP(object):
	
	def __init__(self, n_in, n_out, learning_rate = 0.001, batch_size=16, epochs=100):
		self.learning_rate = learning_rate
		# self.model = MLP2(n_in, n_out)
		self.model = MLP3(n_in, n_out)
		self.batch_size = batch_size
		self.epochs = epochs
	
	def fit(self, x, y):
		tensor_x= torch.from_numpy(x)
		tensor_y= torch.from_numpy(y)

		my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
		my_dataloader = data.DataLoader(my_dataset, batch_size=self.batch_size) # create your dataloader

		criterion = nn.NLLLoss()
		# optimizer = torch.optim.SGD(self.model.parameters()), lr=self.learning_rate)
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)

		for epoch in range(self.epochs):
			for i, (features, labels) in enumerate(my_dataloader):

				# Clear gradients w.r.t. parameters
				optimizer.zero_grad()
				
				# Forward pass to get output/logits
				outputs = self.model(features)
				# print(outputs.shape)
				
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


# trd=np.load("/Users/gexarcha/data/hummus_stdp/569_stdp_output/logistic_tr_set.npy").astype(np.float32)
# trl=np.load("/Users/gexarcha/data/hummus_stdp/569_stdp_output/logistic_tr_label.npy").astype(np.int64)
# ted=np.load("/Users/gexarcha/data/hummus_stdp/569_stdp_output/logistic_te_set.npy").astype(np.float32)
# tel=np.load("/Users/gexarcha/data/hummus_stdp/569_stdp_output/logistic_te_label.npy").astype(np.int64)

trd = np.load("/Users/gexarcha/data/hummus_stdp/0.02_stdp_learning_3class/logistic_tr_set.npy").astype(np.float32)
trl = np.load("/Users/gexarcha/data/hummus_stdp/0.02_stdp_learning_3class/logistic_tr_label.npy").astype(np.int64)
ted = np.load("/Users/gexarcha/data/hummus_stdp/0.02_stdp_learning_3class/logistic_te_set.npy").astype(np.float32)
tel = np.load("/Users/gexarcha/data/hummus_stdp/0.02_stdp_learning_3class/logistic_te_label.npy").astype(np.int64)

dpts = list(range(0,12000,10))


best = 0
bestn = 0
acc = []
for k in dpts[1:]:
	lreg = MLP(n_in=100,n_out=3, epochs=70)
	lreg.fit(trd[-k:,:],trl[-k:])

	acc.append((lreg.predict(ted).numpy()==tel).sum()/tel.shape[0])
	if acc[-1]>best:
		best=acc[-1]
		bestn=k
		print("We have the best test accuracy at {:.05} using {} datapoints  ".format(best,bestn))
