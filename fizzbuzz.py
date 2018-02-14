import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


N_DIGITS = 10
TRAINING_RANGE = range(101, 1024)
TEST_RANGE = range(100)
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DISPLAY_RATE = 100

PASSES = 600


class Net(nn.Module):
    def __init__(self, w_x=N_DIGITS, num_hidden=100, ratio=1):
        super(Net, self).__init__()
        self.num_hidden = num_hidden
        self.main = nn.Sequential(
            nn.Linear(w_x, num_hidden),
            nn.ReLU(True),
            # nn.Linear(num_hidden, num_hidden),
            # nn.ReLU(True),
            nn.Linear(int(num_hidden / ratio), 4)
        )

    def forward(self, x):
        return self.main(x)


def binary_encode(i, num_digits):
    return torch.Tensor([i >> d & 1 for d in range(num_digits)])


def binary_decode(digits):
    return int(sum([(2 ** i) * x for i, x in enumerate(digits)]))


def get_class(i):
    return int(i % 3 == 0) + int(i % 5 == 0) * 2


def weight_init(m):
    if hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.01)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0)


if __name__ == "__main__":
    net = Net()
    net.apply(weight_init)
    training_X = torch.stack([binary_encode(i, N_DIGITS) for i in TRAINING_RANGE], 0)
    # training_Y_OH = torch.LongTensor([(int(not(i % 3 == 0) and not(i % 5 == 0)), int(i % 3 == 0 and not(i % 5 == 0)), int(i % 5 == 0 and not(i % 3 == 0)), int(i % 15 == 0)) for i in TRAINING_RANGE])
    training_Y = torch.LongTensor([[get_class(i)] for i in TRAINING_RANGE])

    dataset = TensorDataset(training_X, training_Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    assert len(training_X) == len(training_Y)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    net.train()
    try:
        for p in range(PASSES):
            for i, data in enumerate(dataloader, 0):
                x, y = data
                x, y = Variable(x), Variable(y.squeeze())

                output = net(x)
                criterion = nn.CrossEntropyLoss()

                optimizer.zero_grad()
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.data)
                if i % DISPLAY_RATE == 0:
                    print("x: {}\ty: {}\tout: {}".format(binary_decode(x[0]), int(y[0].data), output.data.numpy()[0]))
                    print("i: {}\trunning loss: {:.10f}\tloss: {:.10f}" .format(i, running_loss / DISPLAY_RATE, float(loss.data)))
                    running_loss = 0.0
    except KeyboardInterrupt:
        pass

    net.eval()
    true = []
    pred = []
    for i, i_bin in [(i, binary_encode(i, N_DIGITS)) for i in TEST_RANGE]:
        x = Variable(i_bin)
        output = net(x)
        true.append(get_class(i))
        pred.append(np.argmax(output.data.numpy()))
        print("i:{}\ty:{}\typred:{}\tcorrect:{}\t{}".format(i, get_class(i), np.argmax(output.data.numpy()), get_class(i) == np.argmax(output.data.numpy()), output.data.numpy()))

    print("Jaccard score: {}".format(metrics.jaccard_similarity_score(true, pred)))
