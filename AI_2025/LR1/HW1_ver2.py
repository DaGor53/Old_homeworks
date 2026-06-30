import numpy as np

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml

def one_hot_encode(y):
    unique_classes = np.unique(y)
    class_to_index = {classv: indexv for indexv, classv in enumerate(unique_classes)}
    y_indices = np.array([class_to_index[label] for label in y])
    return np.eye(len(unique_classes))[y_indices]

def iris_loader():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    return X,y

def mnist_loader(sample_size=10000):
    test_size=0.2
    random_state=None
    mnist = fetch_openml(name="mnist_784", version=1, as_frame=False, parser="auto")
    X = np.array(mnist.data[:sample_size]) / 255.0
    y = np.array(mnist.target[:sample_size], dtype=int)
    
    return X,y

def map(X, func):
    return np.array([func(x) for x in X])

def shuffle(X, y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    return X, y

def batch(X, y, batch_size):

    X, y = shuffle(X, y)
    
    batches_X = []
    batches_y = []
    
    for start_index in range(0, len(X), batch_size):
        end_index = min(start_index + batch_size, len(X))
        batches_X.append(X[start_index:end_index])  
        batches_y.append(y[start_index:end_index])
    
    return batches_X, batches_y

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    test_size = int(len(X) * test_size)
    indices = np.arange(len(X))

    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_der(x):
    return (x > 0).astype(float)

def linear(x):
    return x

def linear_der(x):
    return np.ones_like(x) 

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_der(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true, y_pred):
    epsilon = 0.000000000001
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_loss_der(y_true, y_pred):
    return y_pred - y_true

def sgd(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

def momentum_sgd(weights, gradient, learning_rate, velocity, momentum=0.9):
    velocity = momentum * velocity - learning_rate * gradient
    return weights + velocity, velocity

def sign_sgd(weights, gradient, learning_rate):
    return weights - learning_rate * np.sign(gradient)

class Layer:
    def __init__(self, input_size, output_size, activation_func, activation_derivative, optimizer):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.optimizer = optimizer

        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weights) + self.biases
        self.output = self.activation_func(self.z)
        return self.output

    def backward(self, d_output, learning_rate, momentum):
        d_activation = d_output * self.activation_derivative(self.z)
        d_weights = np.dot(self.input.T, d_activation)
        d_biases = np.sum(d_activation, axis=0, keepdims=True)
        d_input = np.dot(d_activation, self.weights.T)

        if self.optimizer == momentum_sgd:
            self.weights, self.velocity_w = self.optimizer(self.weights, d_weights, learning_rate, self.velocity_w, momentum)
            self.biases, self.velocity_b = self.optimizer(self.biases, d_biases, learning_rate, self.velocity_b, momentum)
        else:
            self.weights = self.optimizer(self.weights, d_weights, learning_rate)
            self.biases = self.optimizer(self.biases, d_biases, learning_rate)
        
        return d_input

class NeuralNetwork:
    def __init__(self, activation_func=sigmoid, activation_derivative=sigmoid_der, optimizer=sgd, loss_function='mse'):
        self.layers = []
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.optimizer = optimizer

        if loss_function == 'mse':
            self.loss = mse_loss
            self.loss_derivative = mse_loss_der
        elif loss_function == 'cross_entropy':
            self.loss = cross_entropy_loss
            self.loss_derivative = cross_entropy_loss_der

    def add_layer(self, input_size, output_size, activation_func=None, activation_derivative=None):
        activation_func = activation_func if activation_func else self.activation_func
        activation_derivative = activation_derivative if activation_derivative else self.activation_derivative
        self.layers.append(Layer(input_size, output_size, activation_func, activation_derivative, self.optimizer))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_output, learning_rate, momentum):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate, momentum)

    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, momentum=0.9):
        for epoch in range(epochs):
            batches_X, batches_y = batch(X, y, batch_size)
            for batch_X, batch_y in zip(batches_X, batches_y):
                output = self.forward(batch_X)
                error = self.loss_derivative(batch_y,output)
                self.backward(error, learning_rate, momentum)
            
            if epoch % 10 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
