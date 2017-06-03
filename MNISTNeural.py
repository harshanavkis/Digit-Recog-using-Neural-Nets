"""A module to implement a feedforward neural network. It makes use of Stochastic
gradient descent, whose value is computed using the back-propagation algorithm.

Uses python 3.x"""

import random #standard library
import numpy as np #third party library

#define a network class
class Network:
	def __init__(self,sizes):
		self.numOfLayers=len(sizes)
		self.sizes=sizes #has the list containing the number of nodes in each layer
		self.bias=[np.random.randn(y,1) for y in sizes[1:]]#creates a random bias value for each layer except the input
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		#produces the output value given the input a and current values of weights and biases
		for b,w in zip(self.bias,self.weights):
			a=sigmoid(np.dot(w,a)+b)#sigmoid function outputs value between 0 and 1
		return a

	def SGD(self,training_data,iterations,mini_batch_size,eta,test_data=None):
		"""It implements stochastic gradient descent."""
		"""training_data is the list of tuples (x,y),containing inputs and the corresponding 
		outputs. iterations is the number of iterations we train for. mini_batch_size is the 
		reduced size of the training data over which we perform gradient descent"""
		c=list(test_data)
		if test_data: n_test = len(c)
		n=len(training_data)
		for j in range(iterations):
			random.shuffle(training_data)
			mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)#eta is the learning rate
			if test_data:
            			print("Epoch %d: %d / %d"%(j, self.evaluate(test_data), n_test))
            		else:
            			print("Epoch %d complete"%(j))

	def update_mini_batch(self,mini_batch,eta):
		"""uses the bachpropagation algorithm to apply the gradient descent algorithm over the mini_batch
		Then it updates the weights and biases accordingly. eta is the learning rate."""
		nabla_b=[np.zeros(b.shape) for b in self.bias]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_nabla_b,delta_nabla_w=self.backprop(x,y)
			nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
		self.bias=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.bias,nabla_b)]

	def backprop(self,x,y):
		"""Return a tuple (nabla_b,nabla_w) representing the gradient of the cost function
		nabla_w and nabla_b are layer-by-layer lists of numpy arrays"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
        	nabla_w = [np.zeros(w.shape) for w in self.weights]
        	# feedforward
        	activation = x
        	activations = [x] # list to store all the activations, layer by layer
        	zs = [] # list to store all the z vectors, layer by layer
        	for b, w in zip(self.biases, self.weights):
            		z = np.dot(w, activation)+b
            		zs.append(z)
            		activation = sigmoid(z)
            		activations.append(activation)
        	# backward pass
        	delta = self.cost_derivative(activations[-1], y) * \
            		sigmoid_prime(zs[-1])
        	nabla_b[-1] = delta
        	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        	for l in range(2, self.num_layers):
            		z = zs[-l]
            		sp = sigmoid_prime(z)
            		delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            		nabla_b[-l] = delta
            		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        	return (nabla_b, nabla_w)

	def evaluate(self,test_data):
    	"""Returns the number of test inputs for which the neural network predicts the correct
    	output. Output of the neural net is the neuron in the output layer that has the highest 
    	activation value"""
    		test_results=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
    		return sum(int(x==y)for (x,y) in test_results)

    	def cost_derivative(self,output_activations,y):
    	"""return the partial derivative for  the output activations"""
    		return (output_activations-y)

def sigmoid(z):
	#calculates the sigmoid of z
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	#calculates the derivative of the sigmoid function
	return sigmoid(z)*(1-sigmoid(z))
