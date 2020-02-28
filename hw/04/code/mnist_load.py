from mnist import MNIST
import numpy as np
from label_data import PartitionData, LabelData
import time

mndata = MNIST('mnist-data')
Xt, Yt = mndata.load_training()
Xv, Yv = mndata.load_testing()

def load_mnist_data(s0, s1, ratio, usage):
    '''
    Given digits s0, s1, load mnist data into training and validation sets
    :param s0: int[0, 9] First digit to classify
    :param s1:  int[0, 9] Second digit to classify
    :param ratio: float(0, 1) Ratio to put into the training set
    :param usage: float(0, 1) How much of the total MNIST data to use
    :return:
    '''
    X = np.vstack((Xt, Xv))
    Y = np.hstack((Yt, Yv))

    X = X[(Y == s0) | (Y == s1)].astype(np.float32)
    Y = Y[(Y == s0) | (Y == s1)].astype(np.float32)

    M = int(round(np.shape(X)[0] * usage))
    use = np.zeros(np.shape(X)[0], dtype=np.bool)
    use[0:M] = 1.0
    np.random.shuffle(use)

    X = X[use, :]
    Y = Y[use]

    Y = np.array(Y, dtype=np.float32)
    unique = set(np.array(Y, dtype=np.float32))
    Y[Y == min(unique)] = 0.0
    Y[Y == max(unique)] = 1.0

    md = LabelData()
    md.add_data(X, Y)
    mnist_data = PartitionData(md)
    mnist_data.partition(ratio)

    indices_lp = {1.0: max(unique), 0.0: min(unique)}
    return mnist_data, indices_lp

def view_digits(ax, digits, nx, ny):
    '''
    :param ax: pyplot axis object
    :param digits: array [N x 784] MNIST data
    :param nx: number of columns
    :param ny: number of rows
    :return: None
    '''
    width = int(np.sqrt(digits.shape[1]))
    img = np.zeros((nx*width, ny*width))
    idx = 0
    for i in range(0, nx):
        for j in range(0, ny):
            img[i*width:i*width+width, j*width:j*width+width] = digits[idx].reshape((width, width))
            idx += 1
    ax.imshow(img, extent=[0, 1, 0, 1], cmap='Greys')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

def risk(y, yp):
    '''
    :param y: {pm 1} classified values
    :param yp: {pm 1} true values
    :return:
    '''
    return 1/(np.size(y))*np.sum(0.5*np.abs(y - yp))


if __name__ == "__main__":
    usage = 0.1
    ratio = 0.5
    s0, s1 = 0,4
    mnist, names = load_mnist_data(s0, s1, ratio, usage)

