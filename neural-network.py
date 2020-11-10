import numpy as np

# Can be set to make sure the random weights and biases generated are going to be the same for each time the code runs
np.random.seed(0)

# suppress scientific notation
np.set_printoptions(suppress=True)

# input data
input = np.array([[0.5, 0.3, 10.0, 0.2],
                  [0.7, 0.6, 0.25, 0.5],
                  [0.6, 0.5, 12.0, 0.3],
                  [0.5, 0.1, 0.5, 0.12],
                  [0.25, 0.8, 0.6, 0.7]])

# target output
target = np.array([[1,0,1,0,0]]).T

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

class Neural_Network:

    def __init__(self, n_inputs, n_neurons):
        # random weights created, biases = 0
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # multiplies input matrix with weights and adds biases
        self.output = sigmoid(np.dot(inputs, self.weights)+self.biases)

    def backpropagation_hidden_layer(self):
        # backpropagation for hidden layer
        # application of the chain rule to find the partial derivative of the loss function with respect to corresponding weight
        # slope of loss function determines change in weights and biases
        d_weights = (-1) * np.dot(input.T, (np.dot(2 * (layer2.output - target) * sigmoid_derivative(self.output), layer2.weights) * sigmoid_derivative(self.output)))
        d_biases = (-1) * np.sum((self.output - target) * sigmoid_derivative(self.output))
        self.weights += d_weights
        self.biases += d_biases

    def backpropagation_output_layer(self):
        # backpropagation for output layer
        d_weights = (-1) * np.dot(layer1.output.T, ((self.output - target) * sigmoid_derivative(self.output)))
        d_biases = (-1) * np.sum((self.output - target) * sigmoid_derivative(self.output))
        self.weights += d_weights
        self.biases += d_biases

# Neural_Network(number of inputs, number of neurons);
# The number of inputs for layer2 to has to be the same as the number of neurons in layer1
layer1 = Neural_Network(4, 5)
layer2 = Neural_Network(5, 1)

# calls forward function with the original input as input
layer1.forward(input)

# calls forward function with the output of layer1 as input
layer2.forward(layer1.output)

# first try with random weights and biases = 0
print("first try")
print(layer2.output)

# backpropagation to fit weights and biases
layer2.backpropagation_output_layer()
layer1.backpropagation_hidden_layer()

# inputs forwarded with new weights and biases
layer1.forward(input)
layer2.forward(layer1.output)

# second try with updated weights and biases from one backpropagation
print("second try")
print(layer2.output)

# repeat the learning process
for i in range(5000):
    layer2.backpropagation_output_layer()
    layer1.backpropagation_hidden_layer()
    layer1.forward(input)
    layer2.forward(layer1.output)
    print(i)
    print(layer2.output)


# PREDICTIONS

# input to be checked in the end
input = np.array([[0.3, 0.4, 11.0, 0.3],
                  [0.5, 0.35, 0.2, 0.7],
                  [0.45, 0.7, 9.0, 0.6],
                  [0.5, 0.1, 0.7, 0.25]])

# forward input (output is going to be calculated with updated weights and biases)
layer1.forward(input)
layer2.forward(layer1.output)

print("PREDICTIONS")
print("input:")
print(input)
print("output (prediction):")
print(layer2.output)


