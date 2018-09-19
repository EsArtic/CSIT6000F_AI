#!/usr/bin/python

import numpy as np
import math

N = 49
M = 10
LR = 0.1
INITIAL = [0]
DATA_ROOT = './training-set.csv'

def load_data():
    data_input = open(DATA_ROOT, 'r')
    X = []
    labels = []
    for line in data_input:
        items = line.strip().split(',')
        n = len(items)
        x = []
        for i in range(n - 1):
            x.append(float(items[i]))
        x.append(1)
        X.append(x)
        labels.append(int(items[n - 1]))

    return X, labels

def modify_weights(predicts, labels, previous, X):
    diff = predicts - labels
    temp = np.dot(diff.T, X)

    new_weight = previous - (LR * temp.T / M)
    return new_weight

def main():

    X, labels = load_data()
    weights = np.array(INITIAL * M).reshape(M, 1)

    # temp = np.array([1] * len(X))
    # temp = np.transpose(temp)
    # X = np.array(X)
    # X = np.column_stack((X, temp))

    X = np.array(X)
    labels = np.array(labels).reshape(N, 1)

    for i in range(10000):

        # print('X.shape:', X.shape, 'weight.shape:', weights.shape)
        predict_y = np.dot(X, weights)
        # print('predicts.shape', predict_y.shape)

        if i % 1000 == 99:
            diff = predict_y - labels
            RMSE = 0
            for item in diff:
                RMSE += (float(item)) ** 2
            print('Round %d, RMSE = %f' % (i, math.sqrt(RMSE / M)))

        weights = modify_weights(predict_y, labels, weights, X)

    print(weights)
    predict = np.dot(X, weights)
    correct = 0
    for i in range(N):
        result = 0
        if predict[i] >= 0.0:
            result = 1
        else:
            result = 0
        print(result, labels[i])
        if result == labels[i]:
            correct += 1
    print('Accuracy:', float(correct) / N)

if __name__ == '__main__':
    main()