# Neural Network from scratch
A simple neural network from scratch, built in Python.

# How it works
This artificial neural network analyzes an initial input with 4 values per instance (row) in the input dataset.
Each instance is matched with a result.
That result can be either a 0 or a 1 and is listed in the variable 'target'. Those target values represent the values which should be predicted in the future for new inputs. First, however, the neural network has to 'learn' how to predict the correct targets for each instance. This can be done by using the backpropagation algorithm.

The backpropagation algorithm will try to change the weights and biases of the network in a way so that the output (prediction) for each instance gets as close as possible to the corresponding target value. This process represents the 'learning' of the artificial neural network.

After training the model, the results for a new input are going to be predicted (the new input can be found near the end of the code). The new input is going to be run through the neural network with the updated weights and biases.
This time, of course, there is no variable called 'target' as this is the result which what we want to predict.

A predicted result is not going to be a 0 or a 1, but a number between those two numbers.
A result of 0.98, for example, means that the network is predicting the result to be a 1 with a relatively high certainty. A value of 0.57, for example, would mean that the network is still predicting the same result, but with a significantly lower certainty.

# Layers
This neural network consists of 3 layers:
- Input layer
- Hidden layer
- Output layer

# Neurons
The input layer consists of 4 neurons, the hidden layer consists of 5.
In the output layer there is only going to be one neuron which gives us our final output (result).

# Weights and biases
Each layer is connected via weights which are going to be changed automatically during the backpropagation.
The same applies for biases which are added to the neural network in the input and hidden layer

