import numpy as np 

from NN import NN as nn

class get_model(nn):
    def __init__(self, input_channel=1, output_channel=1, hidden_node=100, initialization="Xavier"):
        super(get_model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_node = hidden_node

        # Forget Gate
        # f^{(t)}
        self.weight['U_f'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)
        self.weight['W_f'] = nn.initialize_weights(self.hidden_node, self.hidden_node, initialization)
        self.weight['b_f'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)

        # Input Gate
        # i^{(t)}
        self.weight['U_i'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)
        self.weight['W_i'] = nn.initialize_weights(self.hidden_node, self.hidden_node, initialization)
        self.weight['b_i'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)
        # a^{(t)}
        self.weight['U_a'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)
        self.weight['W_a'] = nn.initialize_weights(self.hidden_node, self.hidden_node, initialization)
        self.weight['b_a'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)

        # Output Gate
        # o^{(t)}
        self.weight['U_o'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)
        self.weight['W_o'] = nn.initialize_weights(self.hidden_node, self.hidden_node, initialization)
        self.weight['b_o'] = nn.initialize_weights(self.hidden_node, self.input_channel, initialization)

        self.weight['V'] = nn.initialize_weights(self.output_channel, self.hidden_node, initialization)
        self.weight['b_V'] = nn.initialize_weights(self.input_channel, self.output_channel, initialization)

    def forward(self, x):
        self.inputs = x

        self.hidden_states = np.zeros((1, self.hidden_node))
        self.C_t = np.zeros((1, self.hidden_node))
        self.a_t = np.zeros((1, self.hidden_node)) # tilde_C_t
        self.f_t = np.zeros((1, self.hidden_node))
        self.o_t = np.zeros((1, self.hidden_node))
        self.i_t = np.zeros((1, self.hidden_node))

        for i in range(len(x)):
            self.y_t, h_t, o_t, C_t, a_t, i_t, f_t = self.cell(x[i].reshape(1,1), self.hidden_states[i].reshape(self.hidden_node, 1), self.C_t[i].reshape(self.hidden_node, 1))
            self.hidden_states = np.append(self.hidden_states, h_t.T, axis=0)
            self.C_t = np.append(self.C_t, C_t.T, axis=0)
            self.a_t = np.append(self.a_t, a_t.T, axis=0)
            self.f_t = np.append(self.f_t, f_t.T, axis=0)
            self.o_t = np.append(self.o_t, o_t.T, axis=0)
            self.i_t = np.append(self.i_t, i_t.T, axis=0)

        return self.y_t
    
    def cell(self, x, h_p, C_p):
        # Forget Gate
        f_t = nn.sigmoid(np.dot(self.weight['W_f'], h_p) + np.dot(self.weight['U_f'], x) + self.weight['b_f'])

        # Input Gate
        i_t = nn.sigmoid(np.dot(self.weight['W_i'], h_p) + np.dot(self.weight['U_i'], x) + self.weight['b_i'])
        a_t = nn.tanh(np.dot(self.weight['W_a'], h_p) + np.dot(self.weight['U_a'], x) + self.weight['b_a'])

        # Cell state update
        C_t = C_p * f_t + i_t * a_t

        # Output Gate
        o_t = nn.sigmoid(np.dot(self.weight['W_o'], h_p) + np.dot(self.weight['U_o'], x) + self.weight['b_o'])
        h_t = o_t * nn.tanh(C_t)

        y_t = np.dot(self.weight['V'], h_t) + self.weight['b_V']

        return y_t, h_t, o_t, C_t, a_t, i_t, f_t
    
    def back_prop(self, y):
        super(get_model, self).back_prop()
        n = len(self.inputs)
        self.grad['b_V'] = self.y_t - y
        self.grad['V'] = np.dot(self.grad['b_V'], self.hidden_states[-1].reshape(1, self.hidden_node))

        delta_h = np.zeros((n+1, self.hidden_node))
        delta_C = np.zeros((n+1, self.hidden_node))
        for i in reversed(range(n)):
            Delta = self.o_t[i+1].reshape(self.hidden_node, 1) * nn.tanh(self.C_t[i+1].reshape(self.hidden_node, 1), der=True)

            Gamma = nn.sigmoid(self.o_t[i+1].reshape(self.hidden_node, 1), der=True) * nn.tanh(self.C_t[i+1].reshape(self.hidden_node, 1)) * self.weight['W_o'].T
            Gamma += Delta * nn.sigmoid(self.f_t[i+1].reshape(self.hidden_node, 1), der=True) * self.C_t[i].reshape(self.hidden_node, 1) * self.weight['W_f'].T
            Gamma += Delta * nn.sigmoid(self.i_t[i+1].reshape(self.hidden_node, 1), der=True) * self.a_t[i+1].reshape(self.hidden_node, 1) * self.weight['W_i'].T
            Gamma += Delta * self.i_t[i+1].reshape(self.hidden_node, 1) * nn.tanh(self.a_t[i+1].reshape(self.hidden_node, 1), der=True) * self.weight['W_a'].T

            delta_h[i] = np.dot(self.weight['V'].T, self.grad['b_V']).T + np.dot(Gamma, delta_h[i].reshape(self.hidden_node, 1)).T
            delta_C[i] = delta_h[i] * self.o_t[i] * nn.tanh(self.C_t[i], der=True) + delta_C[i+1] * self.f_t[i+1]

        # f^{(t)}
        self.grad['U_f'] = np.zeros((self.hidden_node, self.input_channel))
        self.grad['W_f'] = np.zeros((self.hidden_node, self.hidden_node))
        self.grad['b_f'] = (delta_C[1:] * self.C_t[:n] * nn.sigmoid(self.f_t[1:], der=True)).T

        # i^{(t)}
        self.grad['U_i'] = np.zeros((self.hidden_node, self.input_channel))
        self.grad['W_i'] = np.zeros((self.hidden_node, self.hidden_node))
        self.grad['b_i'] = (delta_C[1:] * self.a_t[1:] * nn.sigmoid(self.i_t[1:], der=True)).T
        # a^{(t)}
        self.grad['U_a'] = np.zeros((self.hidden_node, self.input_channel))
        self.grad['W_a'] = np.zeros((self.hidden_node, self.hidden_node))
        self.grad['b_a'] = (delta_C[1:] * self.i_t[1:] * nn.tanh(self.a_t[1:], der=True)).T

        # o^{(t)}
        self.grad['U_o'] = np.zeros((self.hidden_node, self.input_channel))
        self.grad['W_o'] = np.zeros((self.hidden_node, self.hidden_node))
        self.grad['b_o'] = (delta_h[1:] * nn.tanh(self.C_t[1:]) * nn.sigmoid(self.o_t[1:], der=True)).T
        for i in range(1, n+1):
            self.grad['U_f'] += np.dot(self.grad['b_f'][:,i-1].reshape(self.hidden_node, 1), self.inputs[i-1].reshape(1, self.input_channel))
            self.grad['W_f'] += np.dot(self.grad['b_f'][:,i-1].reshape(self.hidden_node, 1), self.hidden_states[i-1].reshape(1, self.hidden_node))

            self.grad['U_i'] += np.dot(self.grad['b_i'][:,i-1].reshape(self.hidden_node, 1), self.inputs[i-1].reshape(1, self.input_channel))
            self.grad['W_i'] += np.dot(self.grad['b_i'][:,i-1].reshape(self.hidden_node, 1), self.hidden_states[i-1].reshape(1, self.hidden_node))
            self.grad['U_a'] += np.dot(self.grad['b_a'][:,i-1].reshape(self.hidden_node, 1), self.inputs[i-1].reshape(1, self.input_channel))
            self.grad['W_a'] += np.dot(self.grad['b_a'][:,i-1].reshape(self.hidden_node, 1), self.hidden_states[i-1].reshape(1, self.hidden_node))

            self.grad['U_o'] += np.dot(self.grad['b_o'][:,i-1].reshape(self.hidden_node, 1), self.inputs[i-1].reshape(1, self.input_channel))
            self.grad['W_o'] += np.dot(self.grad['b_o'][:,i-1].reshape(self.hidden_node, 1), self.hidden_states[i-1].reshape(1, self.hidden_node))
        
        self.grad['U_f'] = np.clip(self.grad['U_f'], -1, 1)
        self.grad['W_f'] = np.clip(self.grad['W_f'], -1, 1)
        self.grad['b_f'] = np.clip(np.sum(self.grad['b_f'], axis=1, keepdims=True), -1, 1)

        self.grad['U_i'] = np.clip(self.grad['U_i'], -1, 1)
        self.grad['W_i'] = np.clip(self.grad['W_i'], -1, 1)
        self.grad['b_i'] = np.clip(np.sum(self.grad['b_i'], axis=1, keepdims=True), -1, 1)

        self.grad['U_a'] = np.clip(self.grad['U_a'], -1, 1)
        self.grad['W_a'] = np.clip(self.grad['W_a'], -1, 1)
        self.grad['b_a'] = np.clip(np.sum(self.grad['b_a'], axis=1, keepdims=True), -1, 1)

        self.grad['U_o'] = np.clip(self.grad['U_o'], -1, 1)
        self.grad['W_o'] = np.clip(self.grad['W_o'], -1, 1)
        self.grad['b_o'] = np.clip(np.sum(self.grad['b_o'], axis=1, keepdims=True), -1, 1)


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