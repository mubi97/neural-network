import numpy as np


X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([[1], [1], [0]])

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def deri_sigmoid(x):
	return x * (1 - x)

epoch = 300
alpha = 0.1

fl_neurons = X.shape[1]
hl_neurons = 3
out_neurons = 1

weight = np.random.uniform(size=(fl_neurons,hl_neurons))
bias = np.random.uniform(size=(1,hl_neurons))
w_out = np.random.uniform(size=(hl_neurons,out_neurons))
b_out = np.random.uniform(size=(1,out_neurons))

for i in xrange(epoch):

	hl_in = sigmoid(np.dot(X, weight) + bias)
	hl_out = sigmoid(np.dot(hl_in, w_out) + b_out)

	E = y - hl_out
	sl_out = deri_sigmoid(hl_out)
	sl_hl  = deri_sigmoid(hl_in)
	d_out = E * sl_out
	e_hl = d_out.dot(w_out.T)
	d_hl = e_hl * sl_hl
	w_out += hl_in.T.dot(d_out) * alpha
	b_out += np.sum(d_out, axis = 0, keepdims = True) * alpha
	weight += X.T.dot(d_hl) * alpha
	bias += np.sum(d_hl, axis = 0, keepdims = True) * alpha


print weight
