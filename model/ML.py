import numpy as np

class ML():
    def __init__(self):
        pass

    @staticmethod
    def tanh(x, der=False):
        if der:
            return 1 - x**2
        else:
            return np.tanh(x)
            
    @staticmethod
    def sigmoid(x, der=False):
        if der:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def loss(output, label):
        return 0.5 * (output - label)**2
    
    @staticmethod
    def Xavier(input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(6/(1+input_size+output_size))
    
    @staticmethod
    def He(input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
    
    @staticmethod
    def Gaussian(input_size, output_size):
        return np.random.randn(input_size, output_size)

    @staticmethod
    def Random(input_size, output_size):
        return np.random.uniform(-1, 1, (input_size, output_size))
    
    @staticmethod
    def Constant0(input_size, output_size):
        return np.zeros(input_size, output_size)

    @staticmethod
    def SGD(weight, gradient, learning_rate):
        weight -= learning_rate * gradient
        return weight, 0, 0
    
    @staticmethod
    def Momentum(weight, gradient, momentum_layer, learning_rate, gamma):
        momentum_layer = gamma * momentum_layer + learning_rate * gradient
        weight -= momentum_layer
        return weight, momentum_layer, 0
    
    @staticmethod
    def AdaGrad(weight, gradient, learning_rate):
        weight -= learning_rate * gradient / np.sqrt(np.sum(gradient**2))
        return weight, 0, 0
    
    @staticmethod
    def RMSProp(weight, gradient, momentum_layer, learning_rate, gamma, eps):
        momentum_layer = gamma * momentum_layer + (1 - gamma) * gradient**2
        weight -= learning_rate * gradient / (np.sqrt(momentum_layer) + eps)
        return weight, momentum_layer, 0
    
    @staticmethod
    def Adam(weight, gradient, momentum_layer, RSM_layer, learning_rate, beta, eps, epoch):
        momentum_layer = beta[0] * momentum_layer + (1 - beta[0]) * gradient
        RSM_layer = beta[1] * RSM_layer + (1 - beta[1]) * gradient**2
        momentum_layer_hat = momentum_layer / (1 - beta[0]**epoch)
        RSM_layer_hat = RSM_layer / (1 - beta[1]**epoch)
        weight -= learning_rate * momentum_layer_hat / (np.sqrt(RSM_layer_hat) + eps)
        return weight, momentum_layer, RSM_layer
    
    @staticmethod
    def AMSGrad(weight, gradient, momentum_layer, RSM_layer, learning_rate, beta, eps):
        momentum_layer = beta[0] * momentum_layer + (1 - beta[0]) * gradient
        RSM_layer_temp = beta[1] * RSM_layer + (1 - beta[1]) * gradient**2
        RSM_layer_hat = np.maximum(RSM_layer, RSM_layer_temp)
        weight -= learning_rate * momentum_layer / (np.sqrt(RSM_layer_hat) + eps)
        return weight, momentum_layer, RSM_layer_temp