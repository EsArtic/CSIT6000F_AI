#!/usr/bin/python

import numpy as np
import random

INITIAL = 1000  # Number of the first generation
RANGE = 10      # Random range
LOOP = 10       # Rounds of creating new generations
N = 10          # Length of feature map + 1 (x1, ..., xn) + threshold
M = 49          # Length of training set

DATA_ROOT = './training-set.csv'

'''
    randomly generate a value between -10.0 and 10.0
'''
def get_rand_weight():
    return random.random() * RANGE * random.uniform(-1.0, 1.0)

'''
    initialize the first generation programs
'''
def init():
    population = []
    for i in range(INITIAL):
        curr = []
        for i in range(N):
            # curr.append(random.random() * RANGE * random.randint(-1, 1))
            curr.append(get_rand_weight())
        population.append(curr)

    return population

'''
    load the training set from file
'''
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

'''
    evaluate current program and return the score
'''
def eval(w, X, labels):
    X = np.array(X)

    w = np.array(w).reshape(N, 1)
    predict = np.dot(X, w)

    count = 0
    for i in range(M):
        temp = 0
        if predict[i] > 0.0:
            temp = 1
        if temp == labels[i]:
            count += 1

    evaluation = float(count) / 49 * 100
    return evaluation

def find_best(curr_generation, X, labels):
    best = curr_generation[0]
    for elem in curr_generation:
        if eval(elem, X, labels) > eval(best, X, labels):
            best = elem

    return best

def tournament_selection(curr_generation, X, labels):
    candidates = []
    for i in range(7):
        candidates.append(curr_generation[random.randint(0, len(curr_generation) - 1)])

    target = find_best(candidates, X, labels)

    return target

def crossover(w1, w2):
    child = w1[:5] + w2[5:]
    return child

def main():
    curr_generation = init()
    X, labels = load_data()

    total = len(curr_generation)
    for i in range(LOOP):
        next_generation = []
        while (len(next_generation) < (total / 10)):
            selected = tournament_selection(curr_generation, X, labels)
            next_generation.append(selected)

        while (len(next_generation) < total):
            father = tournament_selection(curr_generation, X, labels)
            mother = tournament_selection(curr_generation, X, labels)
            child = crossover(father, mother)
            next_generation.append(child)

        for j in range(int(total / 100)):
            next_generation[random.randint(0, len(next_generation) - 1)][random.randint(0, N - 1)] = get_rand_weight()

        curr_generation = next_generation
        best = find_best(curr_generation, X, labels)
        print('Round %d, Curr best performance is: %.2f' % (i + 1, eval(best, X, labels)))

    best = find_best(curr_generation, X, labels)
    predict = np.dot(X, best)
    for i in range(M):
        temp = 0
        if predict[i] > 0.0:
            temp = 1
        
        print(temp, labels[i])

    # for elem in curr_generation:
    #     evaluation = eval(elem, X, labels)
    #     print(evaluation)

if __name__ == '__main__':
    main()