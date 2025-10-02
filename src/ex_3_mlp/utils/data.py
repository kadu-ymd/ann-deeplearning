import numpy as np
import random


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

    def __init__(self, **kwargs):
        self.input  = kwargs.get("input")
        self.output = kwargs.get("output")
        self.W_hidden = kwargs.get("W_hidden")
        self.b_hidden = kwargs.get("b_hidden")
        self.W_output = kwargs.get("W_output")
        self.b_output = kwargs.get("b_output")
        self.eta = kwargs.get("eta", 0.001)

        # Hidden layer
        self.hidden_activation   = kwargs.get("hidden_activation")
        self.hidden_activation_d = kwargs.get("hidden_activation_d")

        # Output layer (opcional)
        self.output_activation   = kwargs.get("output_activation", None)
        self.output_activation_d = kwargs.get("output_activation_d", None)

        # Loss
        self.loss_function   = kwargs.get("loss_function")
        self.loss_function_d = kwargs.get("loss_function_d")

    def forward(self):
        # Hidden layer
        # z1_pre: (n_neurons X n_samples); 
        # W1: (n_neurons X n_feat); input: (n_feat X n_samples); b1: (n_neurons X n_samples)
        z1_pre = self.W_hidden.T @ self.input + self.b_hidden
        z1_act = self.hidden_activation(z1_pre)

        # Output layer
        # z2_pre: (n_outputs X n_samples); 
        # W2: (n_outputs X n_neurons); z1_act: (n_neurons X n_samples); b2: (n_outputs X n_samples)
        z2_pre = self.W_output.T @ z1_act + self.b_output

        if self.output_activation:
            z2_act = self.output_activation(z2_pre)
        else:
            z2_act = z2_pre

        return z1_pre, z1_act, z2_pre, z2_act

    def loss_calculation(self, true_label, predicted_label):
        return self.loss_function(true_label, predicted_label)

    def backpropagation(self, z1_pre, z1_act, z2_pre, z2_act):
        # formato n_output X n_samples
        output_error = self.loss_function_d(self.output, z2_act)

        if self.output_activation_d:
            output_error *= self.output_activation_d(z2_pre) 

        # formato n_neurons X n_samples
        hidden_error = (self.W_output @ output_error) * self.hidden_activation_d(z1_pre)

        # Gradientes
        W_output_gradient = z1_act @ output_error.T
        b_output_gradient = np.sum(output_error, axis=1, keepdims=True)
        W_hidden_gradient = self.input @ hidden_error.T
        b_hidden_gradient = np.sum(hidden_error, axis=1, keepdims=True)

        return W_hidden_gradient, b_hidden_gradient, W_output_gradient, b_output_gradient

    def update_weights(self, W_hidden_gradient, b_hidden_gradient,
                       W_output_gradient, b_output_gradient):
        self.W_hidden -= self.eta * W_hidden_gradient
        self.b_hidden -= self.eta * b_hidden_gradient
        self.W_output -= self.eta * W_output_gradient
        self.b_output -= self.eta * b_output_gradient
        return self.W_hidden, self.b_hidden, self.W_output, self.b_output



def shuffle_sample(sample_array, labels_array):
    lista = list(zip(sample_array, labels_array))
    random.shuffle(lista)

    features, labels = zip(*lista)
    return np.array(features), np.array(labels)


def train_test_split(sample, sample_labels, train_size: float = .8):
    return sample[:int(train_size * len(sample))], sample[:int(train_size * len(sample))], sample_labels[:int(train_size * len(sample_labels))], sample_labels[:int(train_size * len(sample_labels))]


def binary_cross_entropy(y_true, y_pred):
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    VP = np.sum((y_true == 1) & (y_pred == 1))  # verdadeiros positivos
    VN = np.sum((y_true == 0) & (y_pred == 0))  # verdadeiros negativos
    FP = np.sum((y_true == 0) & (y_pred == 1))  # falsos positivos
    FN = np.sum((y_true == 1) & (y_pred == 0))  # falsos negativos

    return np.array([[VN, FP],
                     [FN, VP]])


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
