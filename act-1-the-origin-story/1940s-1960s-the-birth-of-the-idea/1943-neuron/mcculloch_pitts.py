'''
McCulloch & Pitts Neuron
- The first mathematical model of a biological neuron.
- Takes inputs, multiplies by weights (connection strengths),
- sums them up, and fires if the sum meets a threshold.

Limitation: weights must be set by hand.
'''

class McCullochPittsNeuron:
    
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
    
    # step function
    def predict(self, inputs):
        n = len(inputs)
        weighted_sum = 0
        for i in range(n):
            weighted_sum += inputs[i] * self.weights[i]
        if weighted_sum >= self.threshold: # step function
            return 1
        return 0

if __name__ == "__main__":
    
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # AND & OR
    and_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    or_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)

    for input in inputs:
        print(input, 'AND', and_neuron.predict(input), 'OR', or_neuron.predict(input))
    
    # NOT
    # negative weights - that's exactly how some neurons supresses other neurons from firing
    not_neuron = McCullochPittsNeuron(weights=[-1], threshold=0)
    print([0], 'NOT', not_neuron.predict([0]))
    print([1], 'NOT', not_neuron.predict([1]))

    # XOR
    # xor_neuron = McCullochPittsNeuron(weights=[-1, 1], threshold=0)
    # xor_neuron = McCullochPittsNeuron(weights=[-1, 1], threshold=0.1)
    xor_neuron = McCullochPittsNeuron(weights=[-2, 1], threshold=0.5)
    for input in inputs:
        print(input, 'XOR', xor_neuron.predict(input))
    
    # XOR — impossible for a single neuron (not linearly separable)
    # This is what Minsky & Papert proved in 1969
    # Funding dried up - Researchers abandoned neural networks.
    # and what caused the first AI winter - 15 years (working on this was considered career dead end)
    