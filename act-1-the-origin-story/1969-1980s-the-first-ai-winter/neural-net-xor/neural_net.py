'''
Neural Network
- multiple layers of neurons
- everyone knew this was the solution, but no one knew how to train it
- we don't want to manually set weights and biases right ? - we can't we woudl only know intput and output
- it needs to learn
- we can find error for the output, how do we find error for the hidden layers ? Even if we do how do we assign saying this neuron in the hidden layer has contributed more compared to the other neuron.
- credit assignment problem
- solution - backpropagation - calculate output error, propogate error to hidden layer - update weight and bias

- we saw that step function is not differentiable
- we need a differentiable function for backpropagation - activation function
- sigmoid !

- why the inital weights as random ? - to break symmetry
- backpropagation formula - calculate output error, propogate error to hidden layer - update weight and bias
- hidden layer errors are calculated for every output neuron error, and then the sum is multiplied by the sigmoid derivative

- output is not hard and fast 0 or 1 (cause we use activation function and not the step function)
- it's a probabbility - confidence of the output being 1
'''

import math
import random

class NeuralNet:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        # weights
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.output_weights = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        # biases
        self.hidden_bias = [0.0 for _ in range(hidden_size)]
        self.output_bias = [0.0 for _ in range(output_size)]
    
    # activation function
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    # forward pass - prediction
    def forward(self, inputs):
        # hidden layer outputs
        hidden_outputs = []
        for i in range(self.hidden_size):
            weighted_sum = 0
            for j in range(self.input_size):
                weighted_sum += self.hidden_weights[i][j] * inputs[j]
            hidden_output = self.sigmoid(weighted_sum + self.hidden_bias[i])
            hidden_outputs.append(hidden_output)
        
        # output layer outputs
        outputs = []
        for i in range(self.output_size):
            weighted_sum = 0
            for j in range(self.hidden_size):
                weighted_sum += self.output_weights[i][j] * hidden_outputs[j]
            output = self.sigmoid(weighted_sum + self.output_bias[i])
            outputs.append(output)
        
        return (hidden_outputs, outputs)
    
    # TRAIN
    def train(self, training_data, max_epochs):
        for epoch in range(max_epochs):
            for inputs, targets in training_data:
                # Forward pass
                hidden_outputs, outputs = self.forward(inputs)
                
                # Backpropagation

                # calculate output layer errors
                output_errors = [0.0 for _ in range(self.output_size)]
                hidden_errors = [0.0 for _ in range(self.hidden_size)]
                for i in range(self.output_size):
                    sigmoid_derivative = outputs[i] * (1 - outputs[i]) # sigmoid derivative
                    output_errors[i] = (targets[i] - outputs[i]) * sigmoid_derivative
                    # propagate error to all hidden layers from this output neuron
                    for j in range(self.hidden_size):
                        hidden_errors[j] += output_errors[i] * self.output_weights[i][j]
                # * sigmoid derivative for hidden layer - to get complete hidden layer errors
                for j in range(self.hidden_size):
                    sigmoid_derivative = hidden_outputs[j] * (1 - hidden_outputs[j]) # sigmoid derivative
                    hidden_errors[j] *= sigmoid_derivative
                
                # update weights and bias
                for j in range(self.output_size):
                    self.output_bias[j] += self.learning_rate * output_errors[j]
                    for k in range(self.hidden_size):
                        self.output_weights[j][k] += self.learning_rate * output_errors[j] * hidden_outputs[k]
                for j in range(self.hidden_size):
                    self.hidden_bias[j] += self.learning_rate * hidden_errors[j]
                    for k in range(self.input_size):
                        self.hidden_weights[j][k] += self.learning_rate * hidden_errors[j] * inputs[k]

if __name__ == "__main__":

    # XOR training data
    xor_training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    nn = NeuralNet(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
    nn.train(xor_training_data, max_epochs=10000)

    for input, target in xor_training_data:
        _, output = nn.forward(input)
        print(f"Input: {input}, Target: {target}, Predicted: {output}")
    
    # OUTPUT
    # Input: [0, 0], Target: [0], Predicted: [0.01883209287168861]
    # Input: [0, 1], Target: [1], Predicted: [0.9837340144089052]
    # Input: [1, 0], Target: [1], Predicted: [0.9838304893669354]
    # Input: [1, 1], Target: [0], Predicted: [0.016858330636750553]
    # 
    # This basically tells us the model is 98.3 % confident it's a 1, and 1.6% that it's a 1
