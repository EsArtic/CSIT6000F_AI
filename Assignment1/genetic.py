import numpy as np
import random

INITIAL = 100
RANGE = 10
N = 10
M = 49

DATA_ROOT = './training-set.csv'

def init():
    population = []
    for i in range(INITIAL):
        curr = []
        for i in range(N):
            # curr.append(random.random() * RANGE * random.randint(-1, 1))
            curr.append(random.random() * RANGE * random.uniform(-1.0, 1.0))
        population.append(curr)

    return population

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

def crossover(w1, w2):
    child = w1[:5] + w2[5:]
    return child

def main():
    population = init()
    X, labels = load_data()

    print(population[0])
    print(population[1])
    print(crossover(population[0], population[1]))

    # for elem in population:
    #     evaluation = eval(elem, X, labels)
    #     print(evaluation)

if __name__ == '__main__':
    main()