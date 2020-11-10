# Neural Network
A simple neural network from scratch, built in Python.

# Usage
This neural network analyzes an initial input with 4 values per row in the input dataset.
Each row is matched with a result (this is what we want to predict in the end).
That result can be either a 0 or a 1 and is written in the variable 'target'.

After training the model the results for a new input are going to be predicted (you will find that new input near the end of the code).
This time, of course, there is no variable called 'target' as this is what we want to predict.

A predicted result is not going to be a 0 or a 1 but a number between those two numbers.
The algorithm will try to change the weights and biases in a way so that the output (prediction) gets as close as possible to 0 or 1, therefore giving you a hint on how to categorize the row you wanted the prediction for.

# Layers
This neural network includes 3 layers:
- Input layer
- Hidden layer
- Output layer

# Neurons
The input layer consists of 4 neurons, the hidden layer of 5.
In the output layer there is only going to be one neuron which gives us our final output.

# Weights and biases
Each layer is connected via weights which are going to be changed automatically during the backpropagation which resembles the learning process of our neural network.
The same applies for biases which are added to the neural network in the input and hidden layer.

