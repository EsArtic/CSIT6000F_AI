import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

N = 49
M = 10
DATA_ROOT = './training-set.csv'

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()

        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

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

def main():
    X, labels = load_data()

    X = np.array(X).astype(np.float32)
    labels = np.array(labels).reshape(N, 1).astype(np.float32)

    x = torch.from_numpy(X)
    y = torch.from_numpy(labels)

    net = Net(n_feature = 10, n_hidden1 = 2 * M, n_hidden2 = M, n_output = 1)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr = 0.02)
    loss_func = torch.nn.MSELoss()

    for t in range(10000):
        prediction = net(x)

        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 9:
            print('Round %d,' % (t + 1), loss.data.numpy())

    prediction = net(x).data.numpy()
    y = y.data.numpy()
    count = 0
    for i in range(N):
        temp = 0
        if (prediction[i] >= 0.0):
            temp = 1
        if (y[i] == temp):
            count += 1
        print(temp, y[i])
    print('Accuracy = %f' % (float(count) / N))

if __name__ == '__main__':
    main()