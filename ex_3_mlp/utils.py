import numpy as np


class MLP:

    def __init__(self, W, x, bh, V, by, y: float, eta: float):
        self.W = W
        self.x = x
        self.bh = bh
        
        self.V = V
        self.h = 0
        self.by = by

        self.z = 0
        self.u = 0
        
        self.y = y
        self.y_pred = 0

    def forward(self, bh, by, f: function):
        # Hidden layer
        ## Pre-activation
        self.z = np.dot(self.W, self.x) + bh
        
        ## Activation
        self.h = f(self.z)
        
        # Output layer
        ## Pre-activation
        self.u = np.dot(self.V, self.h) + by

        ## Activation
        self.y_pred = f(self.u)

    def loss_function(self):
        return .5 * (self.y - self.y_pred)**2
    
    def backward(self, V: np.array, df: function):
        # Output layer error
        delta_output = (self.y - self.y_pred) * df(self.u)

        # Hidden layer error
        delta_hidden = delta_output * V * df(self.u)

        # Gradient for weights and biases
        output_weight_gradient = delta_output * self.h
        hidden_weight_gradient = delta_hidden * self.x

        output_bias_gradient = delta_output * self.h
        hidden_bias_gradient = delta_hidden

        # Parameter update
        self.V = self.V - self.eta * output_weight_gradient
        self.by = self.by - self.eta * output_bias_gradient

        self.W = self.W - self.eta * hidden_weight_gradient
        self.bh = self.bh - self.eta * hidden_bias_gradient