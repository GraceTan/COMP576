__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def generate_data():
	'''
	generate data
	:return: X: input data, y: given labels
	'''
	np.random.seed(0)
	X, y = datasets.make_moons(200, noise=0.20)
	return X, y

def plot_decision_boundary(pred_func, X, y):
	'''
	plot the decision boundary
	:param pred_func: function used to predict the label
	:param X: input data
	:param y: given labels
	:return:
	'''
	# Set min and max values and give it some padding
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole gid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
	"""
	This class builds and trains a neural network
	"""
	def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
		'''
		:param nn_input_dim: input dimension
		:param nn_hidden_dim: the number of hidden units
		:param nn_output_dim: output dimension
		:param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
		:param reg_lambda: regularization coefficient
		:param seed: random seed
		'''
		self.nn_input_dim = nn_input_dim
		self.nn_hidden_dim = nn_hidden_dim
		self.nn_output_dim = nn_output_dim
		self.actFun_type = actFun_type
		self.reg_lambda = reg_lambda
		
		# initialize the weights and biases in the network
		np.random.seed(seed)
		self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
		self.b1 = np.zeros((1, self.nn_hidden_dim))
		self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
		self.b2 = np.zeros((1, self.nn_output_dim))

	def actFun(self, z, type):
		'''
		actFun computes the activation functions
		:param z: net input
		:param type: Tanh, Sigmoid, or ReLU
		:return: activations
		'''

		# YOU IMPLMENT YOUR actFun HERE
		if type == "tanh":
			return np.tanh(z)
		if type == "sigmoid":
			return 1 / (1 + np.exp(-z)) 
		if type == "relu":
			return np.maximum(0, z)

		return None

	def diff_actFun(self, z, type):
		'''
		diff_actFun compute the derivatives of the activation functions wrt the net input
		:param z: net input
		:param type: Tanh, Sigmoid, or ReLU
		:return: the derivatives of the activation functions wrt the net input
		'''

		# YOU IMPLEMENT YOUR diff_actFun HERE
		if type == "tanh":
			return 1 - np.square(np.tanh(z))
		if type == "sigmoid":
			sigm = self.actFun(z, "sigmoid")
			return sigm * (1 - sigm)
		if type == "relu":
			if z > 0:
				return 1
			return 0

		return None

	def feedforward(self, X, actFun):
		'''
		feedforward builds a 3-layer neural network and computes the two probabilities,
		one for class 0 and one for class 1
		:param X: input data
		:param actFun: activation function
		:return:
		'''
		print("feedforward")
		print(X)
		print("w1: ", self.W1)
		print("b1: ", self.b1)
		print("w2: ", self.W2)
		print("b2: ", self.b2)

		# YOU IMPLEMENT YOUR feedforward HERE

		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = self.actFun(self.z1, self.actFun_type)
		self.z2 = np.dot(self.a1, self.W2) + self.b2
		self.probs = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True) 	# softmax

		print("")
		print("z1")
		print(self.z1)
		print("a1")
		print(self.a1)
		print("z2")
		print(self.z2)
		print("y hat")
		print(self.probs)
		return None

	def calculate_loss(self, X, y):
		'''
		calculate_loss compute the loss for prediction
		:param X: input data
		:param y: given labels
		:return: the loss for prediction
		'''
		num_examples = len(X)
		print("X")
		print(X)
		print("y")
		print(y)
		self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
		# Calculating the loss

		# YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HEREi
		# i don't think this is right
		# data_loss = (-1. / num_examples) * (np.sum(np.sum(y) * np.log(self.probs)))
		data_loss =  - np.sum(np.log(self.probs[range(num_examples), y])) / num_examples

		# Add regulatization term to loss (optional)
		data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
		return (1. / num_examples) * data_loss

	def predict(self, X):
		'''
		predict infers the label of a given data point X
		:param X: input data
		:return: label inferred
		'''
		self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
		return np.argmax(self.probs, axis=1)

	def backprop(self, X, y):
		'''
		backprop run backpropagation to compute the gradients used to update the parameters in the backward step
		:param X: input data
		:param y: given labels
		:return: dL/dW1, dL/b1, dL/dW2, dL/db2
		'''

		# IMPLEMENT YOUR BACKPROP HERE
		num_examples = len(X)
		# dL_da2 = (-1. / num_examples) * (OneHotEncoder(n_values=2, sparse=False).fit_transform(y.reshape(-1, 1))) / self.probs
		# print("dl da2", dL_da2)

		dL_dz2 = self.probs
		dL_dz2[range(num_examples), y] -= 1
		dL_dz2 /= num_examples

		dz2_dW2 = self.a1 

		da2_dz1 = self.diff_actFun(self.z1, self.actFun_type)
		dz1_dW1 = X

		dW2 = np.dot(dz2_dW2.T, dL_dz2)	# dL / dW2
		db2 = np.sum(dL_dz2)	# dL / db2
		dW1 = np.dot((np.dot(dL_dz2, self.W2.T) * da2_dz1).T, X).T	# dL / dW1
		db1 = np.sum(np.dot(dL_dz2, self.W2.T) * da2_dz1)							# dL / dzb1
		return dW1, dW2, db1, db2

	def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
		'''
		fit_model uses backpropagation to train the network
		:param X: input data
		:param y: given labels
		:param num_passes: the number of times that the algorithm runs through the whole dataset
		:param print_loss: print the loss or not
		:return:
		'''
		print("fit model")
		# Gradient descent.
		for i in range(0, num_passes):
			# Forward propagation
			self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
			# Backpropagation
			dW1, dW2, db1, db2 = self.backprop(X, y)

			# Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
			dW2 += self.reg_lambda * self.W2
			dW1 += self.reg_lambda * self.W1

			# Gradient descent parameter update
			self.W1 += -epsilon * dW1
			self.b1 += -epsilon * db1
			self.W2 += -epsilon * dW2
			self.b2 += -epsilon * db2

			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			if print_loss and i % 1000 == 0:
				print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

	def visualize_decision_boundary(self, X, y):
		'''
		visualize_decision_boundary plot the decision boundary created by the trained network
		:param X: input data
		:param y: given labels
		:return:
		'''
		plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():
	# # generate and visualize Make-Moons dataset
	X, y = generate_data()
	plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
	print(X)
	print(y)
	print("")
	# plt.show()

	model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
	print("made model")
	model.fit_model(X,y)
	print("fitted model")
	model.visualize_decision_boundary(X,y)
	print("visual decision boundary")

if __name__ == "__main__":
	main()