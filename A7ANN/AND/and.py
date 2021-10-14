import numpy
from matplotlib import pyplot as plt 

def update(param, gradients, alpha): 
	param["w1"] = param["w1"] - alpha * gradients["dW1"] 
	param["w2"] = param["w2"] - alpha * gradients["dW2"] 
	param["b1"] = param["b1"] - alpha * gradients["db1"] 
	param["b2"] = param["b2"] - alpha * gradients["db2"] 
	return param

def backProp(X_train, Y, cache): 
	m = X_train.shape[1] 
	(Z1, A1, w1, b1, Z2, A2, w2, b2) = cache 
	
	dZ2 = A2 - Y 
	dW2 = numpy.dot(dZ2, A1.T) / m 
	db2 = numpy.sum(dZ2, axis = 1, keepdims = True) 
	
	dA1 = numpy.dot(w2.T, dZ2) 
	dZ1 = numpy.multiply(dA1, A1 * (1- A1)) 
	dW1 = numpy.dot(dZ1, X_train.T) / m 
	db1 = numpy.sum(dZ1, axis = 1, keepdims = True) / m 
	
	gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2, 
				"dZ1": dZ1, "dW1": dW1, "db1": db1} 
	return gradients

def sigmoid(z): 
	return 1 / (1 + numpy.exp(-z)) 

def forwProp(X_train, Y, param): 
	m = X_train.shape[1] 
	w1 = param["w1"] 
	w2 = param["w2"] 
	b1 = param["b1"] 
	b2 = param["b2"] 

	Z1 = numpy.dot(w1, X_train) + b1 
	A1 = sigmoid(Z1) 
	Z2 = numpy.dot(w2, A1) + b2 
	A2 = sigmoid(Z2) 

	cache = (Z1, A1, w1, b1, Z2, A2, w2, b2) 
	logprobs = numpy.multiply(numpy.log(A2), Y) + numpy.multiply(numpy.log(1 - A2), (1 - Y)) 
	cost = -numpy.sum(logprobs) / m 
	return cost, cache, A2

def annNOR(epoch, X_train, Y, param, losses, alpha):
    for i in range(epoch): 
	    losses[i, 0], cache, A2 = forwProp(X_train, Y, param) 
	    gradients = backProp(X_train, Y, cache) 
	    param = update(param, gradients, alpha) 

def initial(inputs, neuron, output): 
	w1 = numpy.random.randn(neuron, inputs) 
	w2 = numpy.random.randn(output, neuron) 
	b1 = numpy.zeros((neuron, 1)) 
	b2 = numpy.zeros((output, 1)) 
	
	param = {"w1" : w1, "b1": b1, 
				"w2" : w2, "b2": b2} 
	return param

X_train = numpy.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = numpy.array([[0, 0, 0, 1]])

neuron = 2
inputs = X_train.shape[0]
output = Y.shape[0]
param = initial(inputs, neuron, output) 
epoch = 100000
alpha = 0.01
losses = numpy.zeros((epoch, 1)) 

print("-----------------Training-------------------")
annNOR(epoch, X_train, Y, param, losses, alpha)
print("Parameters :\n", param)

print("\n-----------------Testing-------------------")
X_test = numpy.array([[0, 0, 1, 1], [1, 0, 1, 0]])
cost, _, A2 = forwProp(X_test, Y, param) 
prediction = (A2 > 0.5) * 1.0
# print(A2)
print("X_test :\n", X_test)
print("Y :", prediction) 

plt.figure() 
plt.plot(losses, color="red") 
plt.xlabel("iteration") 
plt.ylabel("Cost") 
plt.show() 
