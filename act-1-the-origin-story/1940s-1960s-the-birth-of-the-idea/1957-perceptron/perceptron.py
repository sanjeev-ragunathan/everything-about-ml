'''
Perceptron
- Neuron + learn
- learning: adjust weight and bias based on error (target - predicted)
- bias: tells how biased is the neuron to fire or not fire, basically just threshold moved to the left in the predict formula of neuron

- weights is for every input  
- bias is for the whole neuron - added to weighted sum for a particular input
'''

class Perceptron:

    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = [0.0] * self.input_size # random weights
        self.bias = 0.0 # random bias
    
    def predict(self, inputs):
        weighted_sum = 0
        for i in range(self.input_size):
            weighted_sum += inputs[i] * self.weights[i]
        if weighted_sum + self.bias >= 0: # updated formula - step function
            return 1
        return 0
    
    def train(self, training_data, max_epochs):
        # epoch - one full pass through the training data
        for epoch in range(max_epochs):
            all_correct = True
            for input, target in training_data:
                predicted = self.predict(input)
                error = target - predicted
                if error:
                    all_correct = False
                    # update weights - for each input
                    for i in range(len(input)):
                        self.weights[i] += self.learning_rate * error * input[i]
                    # update bias - for neuron
                    self.bias += self.learning_rate * error
            if all_correct:
                print(f"Training converged after {epoch+1} epochs.")
                break
        
        if not all_correct:
            print("Did not converge after max epochs.")





if __name__ == "__main__":

    print()

    # AND training data
    and_training_data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]
    # AND Perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    print("Before training:")
    for input, target in and_training_data:
        predicted = perceptron.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    # Train
    perceptron.train(and_training_data, max_epochs=100)
    print("After training:")
    for input, target in and_training_data:
        predicted = perceptron.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    
    print()

    # OR training data
    or_training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    # OR Perceptron
    perceptron_or = Perceptron(input_size=2, learning_rate=0.3)
    print("Before training OR:")
    for input, target in or_training_data:
        predicted = perceptron_or.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    # Train OR
    perceptron_or.train(or_training_data, max_epochs=100)
    print("After training OR:")
    for input, target in or_training_data:
        predicted = perceptron_or.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    
    print()

    # NOT training data
    not_training_data = [
        ([0], 1),
        ([1], 0)
    ]
    perceptron_not = Perceptron(input_size=1, learning_rate=0.5)
    print("Before training NOT:")
    for input, target in not_training_data:
        predicted = perceptron_not.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    # Train NOT
    perceptron_not.train(not_training_data, max_epochs=100)
    print("After training NOT:")
    for input, target in not_training_data:
        predicted = perceptron_not.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    
    print()

    # XOR training data - not linearly separable
    xor_training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    perceptron_xor = Perceptron(input_size=2, learning_rate=0.1)
    print("Before training XOR:")
    for input, target in xor_training_data:
        predicted = perceptron_xor.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    # Train XOR
    perceptron_xor.train(xor_training_data, max_epochs=100) # will not converge
    print("After training XOR:")
    for input, target in xor_training_data:
        predicted = perceptron_xor.predict(input)
        print(f"Input: {input}, Target: {target}, Predicted: {predicted}")
    
