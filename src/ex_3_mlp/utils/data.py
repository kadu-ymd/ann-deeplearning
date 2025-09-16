import numpy as np


class Data:
    def __init__(self, mu, std, n):
        self.mu_x, self.mu_y = mu
        self.std_x, self.std_y = std
        self.n = n

    def sample_initialize(self) -> tuple[np.ndarray, np.ndarray]:
        return np.random.normal(self.mu_x, self.std_x, self.n), np.random.normal(self.mu_y, self.std_y, self.n)

class MultiDimensionData:
    def __init__(self, mu: list, cov: list, n: int):
        self.mu = np.array(mu)
        self.cov = np.array(cov)
        self.n = n

    def sample_initialize(self):
        return np.random.multivariate_normal(self.mu, self.cov, self.n)


class MLP:

    def __init__(self, input, output, W_hidden, b_hidden, W_output, b_output,
                 eta, activation_function, activation_function_d,
                 loss_function, loss_function_d):
        self.input = input
        self.output = output
        self.W_hidden = W_hidden
        self.b_hidden = b_hidden
        self.W_output = W_output
        self.b_output = b_output
        self.eta = eta
        self.activation_function = activation_function
        self.activation_function_d = activation_function_d
        self.loss_function = loss_function
        self.loss_function_d = loss_function_d

    def forward(self):
        # Hidden Layer
        z1_pre = np.dot(self.W_hidden, self.input.T) + self.b_hidden
        z1_activation = self.activation_function(z1_pre)

        # Output Layer
        z2_pre = np.dot(self.W_output, z1_activation) + self.b_output
        z2_activation = self.activation_function(z2_pre)

        return z1_pre, z1_activation, z2_pre, z2_activation

    def loss_calculation(self, output, prediction):
        return self.loss_function(output, prediction)

    def backpropagation(self, z1_pre, z1_activation, z2_pre, z2_activation):
        # Output layer error
        output_error = self.loss_function_d(self.output, z2_activation) * self.activation_function_d(z2_pre)

        # Hidden layer error
        hidden_error = output_error * self.activation_function_d(z1_pre)

        # Gradients for output layer
        W_output_gradient = output_error * z1_activation.T
        b_output_gradient = output_error

        # Gradients for hidden layer
        W_hidden_gradient = hidden_error * self.input
        b_hidden_gradient = hidden_error

        return W_hidden_gradient, b_hidden_gradient, W_output_gradient, b_output_gradient
    
    def update_weights(self, W_hidden_gradient, b_hidden_gradient, W_output_gradient, b_output_gradient):
        self.W_hidden -= self.eta * W_hidden_gradient
        self.b_hidden -= self.eta * b_hidden_gradient
        self.W_output -= self.eta * W_output_gradient
        self.b_output -= self.eta * b_output_gradient

        return self.W_hidden, self.b_hidden, self.W_output, self.b_output