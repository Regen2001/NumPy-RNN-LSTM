import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import importlib
import argparse

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def main(args):
    model = importlib.import_module(args.model)

    classifier = model.get_model(1, 1, 100, args.initialization)
    classifier.create_optimizer(args.optimizer, args.learning_rate, args.gamma, args.beta)

    x, y = dataset()
    x_test, y_test = dataset(350)
    x_test = x_test[300:]
    y_test = y_test[300:]
    LOSS = np.array([0])
    VAILD_LOSS = np.array([0])
    total_loss= np.array([10])
    total_vaild_loss= np.array([10])
    patience_counter = 0

    for epoch in range(args.epoch):
        print('This is epoch ', epoch+1)
        loss, vaild_loss = train(classifier, x, y, epoch+1)
        outputs = test(classifier, x_test)
        total_loss = np.append(total_loss, np.array([np.mean(np.abs(y_test - outputs))]), axis=0)
        total_vaild_loss = np.append(total_loss, np.array([np.mean(vaild_loss)]), axis=0)
        LOSS = np.append(LOSS, loss, axis=0)
        VAILD_LOSS = np.append(VAILD_LOSS, vaild_loss, axis=0)
        if ((total_loss[epoch+1] - total_loss[epoch]) > args.min_increase) or ((total_vaild_loss[epoch+1] - total_vaild_loss[epoch]) > args.min_increase):
            patience_counter += 1
        if patience_counter == args.max_patience:
            print('Stop trainning')
            break

    LOSS = np.asanyarray(LOSS)
    VAILD_LOSS = np.asanyarray(VAILD_LOSS)
    classifier.save_model('rnn_test.npy')

    plt.figure(dpi=120)
    plt.plot(total_loss)
    plt.figure(dpi=120)
    plt.plot(LOSS)
    plt.figure(dpi=120)
    plt.plot(VAILD_LOSS)
    plt.figure(dpi=120)
    plt.plot([i for i in range(len(x_test))],y_test,outputs.reshape(y_test.shape))
    plt.show()
    return 0

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--model', default='RNN', help='model name [default: RNN')
    parser.add_argument('--initialization', default='He', help='initialization name [default: He')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='RMSProp', help='optimizer for training')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--beta', type=tuple, default=(0.9, 0.999), help='gamma')
    parser.add_argument('--max_patience', type=int, default=3, help='max patience')
    parser.add_argument('--min_increase', type=float, default=-0.1, help='min increase')
    return parser.parse_args()

def train(classifier, x, y, epoch):
    over_loss = []
    vaild_loss = []

    x_vaild = x[250:]
    y_vaild = y[250:]
    x = x[:250]
    y = y[:250]

    print('Training')
    for i in tqdm(range(x.shape[0])):
        pred = classifier(x[i])
        lr = classifier.backward(y[i], epoch=epoch)
        over_loss.append(np.squeeze(classifier.loss(pred, y[i]) / x.shape[0]))
    print('Vaild set Training')
    for j in tqdm(range(x_vaild.shape[0])):
        pred = classifier(x[j])
        lr = classifier.backward(y_vaild[j], epoch=epoch)
        vaild_loss.append(np.squeeze(classifier.loss(pred, y_vaild[j]) / x_vaild.shape[0]))

    return np.asanyarray(over_loss), np.asanyarray(vaild_loss)

def test(classifier, x):
    outputs = []
    print('Testing')
    for i in tqdm(range(len(x))):
        pred = classifier(x[i])
        outputs.append(pred)
    return np.asanyarray(outputs)

def dataset(size=300, timesteps=25):
    x, y = [], []
    sin_wave = np.sin(5*np.arange(size))
    for step in range(sin_wave.shape[0]-timesteps):
        x.append(sin_wave[step:step+timesteps])
        y.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)

if __name__ == '__main__':
    args = parse_args()
    main(args)