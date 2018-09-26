#!/usr/bin/python

import sys
import numpy as np
import random

INITIAL = 2000  # Number of the first generation
RANGE = 1       # Random range
LOOP = 10       # Rounds of creating new generations
SCALE = 7       # Numbers of candidates in tournament selection

# DATA_ROOT = './training-set.csv'


'''
    Randomly generate a value between -RANGE and RANGE
'''
def get_rand_weight():
    return random.uniform(-1.0 * RANGE, RANGE)

'''
    Initialize the first generation of programs
'''
def init(N):
    population = []
    for i in range(INITIAL):
        curr = []
        for i in range(N):
            curr.append(get_rand_weight())
        population.append(curr)

    return population

'''
    Load the training set from file
'''
def load_data(DATA_ROOT):
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

    N = len(X[0]) # Length of feature map + 1 (x1, ..., xn) + threshold
    data_input.close()
    return X, labels, N

'''
    Fitness function for the genetic program
    Using numpy to accelerate the computation
'''
def fitness_func(w, X, labels):
    inputs = np.array(X)
    weights = np.array(w).reshape(len(w), 1)
    predict = np.dot(inputs, weights)

    correct_count = 0
    for i in range(len(X)):
        temp = 0
        if predict[i] >= 0.0:
            temp = 1
        if temp == labels[i]:
            correct_count += 1

    evaluation = float(correct_count) / len(X) * 100
    return evaluation

'''
    Find the best program among the given list
'''
def best(container, X, labels):
    selected = container[0]
    for elem in container:
        if (fitness_func(elem, X, labels) > fitness_func(selected, X, labels)):
            selected = elem

    return selected

def tournament_selection(curr_generation, X, labels):
    candidates = []
    for i in range(SCALE):
        candidates.append(curr_generation[random.randint(0, len(curr_generation) - 1)])

    selected = best(candidates, X, labels)
    return selected

'''
    The crossover operation
'''
def crossover(w1, w2):
    cut_off_point = random.randint(1, 9)
    child = w1[:cut_off_point] + w2[cut_off_point:]
    return child

def produce_next_generation(curr_generation, X, labels):
    next_generation = []

    # Copy 10% of the programs to next generation
    while (len(next_generation) < len(curr_generation) / 10):
        selected = tournament_selection(curr_generation, X, labels)
        next_generation.append(selected)

    # apply crossover operation to produce the rest 90% of the programs
    while (len(next_generation) < len(curr_generation)):
        father = tournament_selection(curr_generation, X, labels)
        mother = tournament_selection(curr_generation, X, labels)
        child = crossover(father, mother)
        next_generation.append(child)

    # randomly select 1% of the programs to do mutation operation
    for i in range(int(len(curr_generation) / 100)):
        selected = random.randint(0, len(next_generation) - 1)
        position = random.randint(0, len(next_generation[selected]) - 1)
        next_generation[selected][position] = get_rand_weight()

    return next_generation

def write_result(output_path, w, X, labels):
    inputs = np.array(X)
    weights = np.array(w).reshape(len(w), 1)
    predict = np.dot(inputs, weights)

    result = open(output_path, 'w')
    for i in range(9):
        result.write('x' + str(i + 1) + ',')
    result.write('label,output\n')

    for i in range(len(X)):
        x = X[i]
        for v in x[: len(x) - 1]:
            result.write(str(v) + ',')
        result.write(str(labels[i]) + ',')
        if predict[i] > 0.0:
            result.write('1\n')
        else:
            result.write('0\n')

    result.close()

def main():
    DATA_ROOT = sys.argv[1]
    X, labels, N = load_data(DATA_ROOT)
    curr_generation = init(N)

    for i in range(LOOP):
        curr_generation = produce_next_generation(curr_generation, X, labels)

        curr_best = best(curr_generation, X, labels)
        accuracy = fitness_func(curr_best, X, labels)
        print('Round %d, Current best performance is: Accuracy = %.2f' % (i + 1, accuracy))
        if accuracy > 99:
            print("The performance already meets the requirement, terminate the evolution.")
            break

    final_best = best(curr_generation, X, labels)
    print('Weights:', final_best)
    print('Accuracy = %.2f' % fitness_func(final_best, X, labels))
    write_result('./output.csv', final_best, X, labels)

if __name__ == '__main__':
    main()