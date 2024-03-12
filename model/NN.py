import numpy as np
from ML import ML as ml

class NN(ml):
    def __init__(self):
        super(NN, self).__init__()
        self.weight = {}
        self.create_optimizer()
    
    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        weight = ''
        for key in self.weight:
            weight += key + ': ' + str(self.weight[key].shape) + '\n'
        return weight

    def forward(self, x):
        pass

    def back_prop(self, y=0):
        self.grad = {}
        pass

    def backward(self, y, epoch=1):
        self.back_prop(y)
        self.update_weight(epoch)

    def update_weight(self, epoch=1):
        if  not ('self.momentum_layer' in vars()):
            self.momentum_layer = {}
            for key in self.weight:
                self.momentum_layer[key] = 0
        if not ('self.RSM_layer' in vars()):
            self.RSM_layer = {}
            for key in self.weight:
                self.RSM_layer[key] = 0

        for key in self.weight:
            self.weight[key], self.momentum_layer[key], self.RSM_layer[key] = NN.update(self.weight[key], self.grad[key], self.momentum_layer[key], self.RSM_layer[key], self.optimizer, epoch)

    def create_optimizer(self, optimizer='SGD', lr=0.01, gamma=0.9, beta=(0.9,0.99), eps=1e-08):
        self.optimizer = {
            'optimizer': optimizer,
            'lr': lr,
            'gamma': gamma,
            'beta': beta,
            'eps': eps
        }

    def save_model(self, file_name):
        np.save(file_name, self.weight)

    def load_model(self, file_name):
        self.weight = np.load(file_name, allow_pickle=True).item()

    @staticmethod
    def initialize_weights(input_size, output_size, initialization='Xavier'):
        if initialization == 'Xavier':
            return ml.Xavier(input_size, output_size)
        elif initialization == 'He':
            return ml.He(input_size, output_size)
        elif initialization == 'Gaussian':
            return ml.Gaussian(input_size, output_size)
        elif initialization == 'Random':
            return ml.Random(input_size, output_size)
        elif initialization == 'Constant0':
            return ml.Constant0(input_size, output_size)
        else:
            raise NotImplemented

    @staticmethod
    def update(weight, gradient, momentum_layer, RSM_layer, optimizer, epoch):
        if optimizer['optimizer'] == 'SGD':
            return ml.SGD(weight, gradient, optimizer['lr'])
        elif optimizer['optimizer'] == 'Momentum':
            return ml.Momentum(weight, gradient, momentum_layer, optimizer['lr'], optimizer['gamma'])
        elif optimizer['optimizer'] == 'AdaGrad':
            return ml.AdaGrad(weight, gradient, optimizer['lr'])
        elif optimizer['optimizer'] == 'RMSProp':
            return ml.RMSProp(weight, gradient, momentum_layer, optimizer['lr'], optimizer['gamma'], optimizer['eps'])
        elif optimizer['optimizer'] == 'Adam':
            return ml.Adam(weight, gradient, momentum_layer, RSM_layer, optimizer['lr'], optimizer['beta'], optimizer['eps'], epoch)
        elif optimizer['optimizer'] == 'AMSGrad':
            return ml.AMSGrad(weight, gradient, momentum_layer, RSM_layer, optimizer['lr'], optimizer['beta'], optimizer['eps'])
        else:
            raise NotImplemented