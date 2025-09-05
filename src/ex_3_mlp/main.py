import numpy as np

# Input vector
x = np.array([[ .5],
              [-.2]])

# Output vector
y = .8

# Hidden layer weights (W)
W = np.array([[.3, -.1],
               [.2,  .4]])

# Hidden layer biases (bh)
bh = np.array([[ .1],
               [-.2]])

# Output layer weights (V)
V = np.array([[ .5, -.3]])

# Output layer biases (by)
by = .2

# Learning rate
eta =.3

# Activation function (hyperbolic tangent)
tanh = lambda x: (np.exp(2*x) - 1)/(np.exp(2*x) + 1)

def main():
    # Forward pass
    ## Hidden layer
    z = np.dot(W, x) + bh # Pre-activation
    h = tanh(z) # Activation

    ## Output layer
    u = np.dot(V, h) + by # Pre-activation
    y_pred = tanh(u)[0][0] # Activation 

    # Loss calculation
    L = .5 * (y - y_pred)**2

    # Backward pass (backpropagation)
    # Start with dL_dy_pred
    dL_dy_pred = (y - y_pred)

    ## . Output layer error (dL_dz2)
    delta_output = dL_dy_pred * (1 - y_pred**2)

    # . Gradients for output layer
    ## Weights (dL_dW2)
    V_gradient = delta_output * h

    ## Bias (dL_db2)
    by_gradient = delta_output

    # . Propagate do hidden layer
    dL_da1 = delta_output * V

    delta_hidden = dL_da1 * (1 - y_pred**2) # dL_dz1

    # . Gradients for hidden layer
    ## Weights (dL_dW1)
    W_gradient = delta_hidden * x

    ## Bias (dL_db1)
    bh_gradient = delta_hidden

    # Parameter update
    new_V = V - eta * V_gradient
    new_by = by - eta * by_gradient
    new_W = W - eta * W_gradient
    new_bh = bh - eta * bh_gradient

    print(new_V)


    return 0

if __name__ == "__main__":
    main()