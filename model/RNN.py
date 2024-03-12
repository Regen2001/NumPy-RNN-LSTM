import numpy as np 
from NN import NN as nn

class get_model(nn):
    def __init__(self, input_channel=1, output_channel=1, hidden_node=100, initialization="Xavier"):
        super(get_model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_node = hidden_node

        self.weight['W_ax'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization=initialization)
        self.weight['W_aa'] = nn.initialize_weights(self.hidden_node, self.hidden_node, initialization=initialization)
        self.weight['W_ya'] = nn.initialize_weights(self.output_channel, self.hidden_node, initialization=initialization)
        self.weight['b_a'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization=initialization)
        self.weight['b_y'] = nn.initialize_weights(self.input_channel, self.output_channel, initialization=initialization)

    def forward(self, x):
        self.inputs = x
        self.hidden_states = np.zeros((1, self.hidden_node))
        for i in range(len(x)):
            h_t, self.y_t = self.cell(x[i].reshape(1,1), self.hidden_states[i].reshape(self.hidden_node, 1))
            self.hidden_states = np.append(self.hidden_states, h_t.T, axis=0)
        return self.y_t
    
    def cell(self, x, h):
        h_t = nn.tanh(np.dot(self.weight['W_ax'], x) + np.dot(self.weight['W_aa'], h) + self.weight['b_a'])
        y_t = np.dot(self.weight['W_ya'], h_t) + self.weight['b_y']
        return h_t, y_t

    # https://zhuanlan.zhihu.com/p/61472450
    def back_prop(self, y):
        super(get_model, self).back_prop()
        n = len(self.inputs)
        self.grad['b_y'] = self.y_t - y
        self.grad['b_a'] = nn.tanh(self.hidden_states[-1].reshape(self.hidden_node, 1), der=True) * np.dot(self.grad['b_y'], self.weight['W_ya']).T

        self.grad['W_ya'] = self.grad['b_y'] * self.hidden_states[-1].reshape(1, self.hidden_node)
        self.grad['W_aa'] = nn.tanh(self.hidden_states[-1].reshape(self.hidden_node, 1), der=True) * np.dot(self.grad['b_y'], self.weight['W_ya']).T
        for i in reversed(range(n)):
            d_h = np.dot(self.weight['W_ya'].T, self.grad['b_y']) + np.dot(self.weight['W_aa'], self.grad['b_a'][:,0].reshape(self.hidden_node, 1)) * self.tanh(self.hidden_states[i].reshape(self.hidden_node, 1), der=True)
            self.grad['b_a'] = np.insert(self.grad['b_a'], 0, d_h.T, axis=1)

        self.grad['W_ax'] = np.zeros(self.weight['W_ax'].shape)
        self.grad['W_aa'] = np.zeros(self.weight['W_aa'].shape)
        for i in reversed(range(n)):
            self.grad['W_ax'] += np.dot(self.grad['b_a'][:,i].reshape(self.hidden_node, 1), self.inputs[i].reshape(1,1))
            self.grad['W_aa'] += np.dot(self.grad['b_a'][:,i].reshape(self.hidden_node, 1), self.hidden_states[i-1].reshape(1, self.hidden_node))
        
        self.grad['b_a'] = np.sum(self.grad['b_a'] ,axis=1, keepdims=True)

        self.grad['b_y'] = np.clip(self.grad['b_y'], -1, 1)
        self.grad['W_ya'] = np.clip(self.grad['W_ya'], -1, 1)
        self.grad['W_ax'] = np.clip(self.grad['W_ax'], -1, 1)
        self.grad['W_aa'] = np.clip(self.grad['W_aa'], -1, 1)
        self.grad['b_a'] = np.clip(self.grad['b_a'], -1, 1)

if __name__ == '__main__':
    x = np.random.randn(25,1)
    y = np.random.randn(1,1)
    model = get_model()
    # print(model)
    pred = model(x)
    loss = model.loss(pred, y)
    model.back_prop(y)
    lr = model.update_weight()
    print(y)
    print(pred)
    print(loss)