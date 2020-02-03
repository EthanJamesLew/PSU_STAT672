from mnist import MNIST
import numpy as np
from label_data import PartitionData, LabelData
import time
from krr import KernelRidgeRegression

mndata = MNIST('mnist')
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

def classify_test(s0, s1, usage, ratio, k):
    '''
    :param s0: First digit to classify
    :param s1: Second digit to classify
    :param usage: float(0, 1) How much of the total MNIST data to use
    :param ratio: float(0, 1) Ratio to put into the training set
    :param k: kernel function
    :return: test results
    '''
    mnist, names = load_mnist_data(s0, s1, ratio, usage)

    ld = LabelData()
    ld.add_data(mnist.training[0], mnist.training[1])

    t0 = time.time()
    kregr = KernelRidgeRegression(ld, k=k, l=.0001)
    t1 = time.time()
    ttotal = t1 - t0

    t0 = time.time()
    y_v = kregr(mnist.validation[0])
    y_v[y_v > 0.5] = 1
    y_v[y_v < 0.5] = -1
    t1 = time.time()
    vtotal = t1 - t0

    t0 = time.time()
    y_t = kregr(mnist.training[0])
    y_t[y_t > 0.5] = 1
    y_t[y_t < 0.5] = -1
    t1 = time.time()
    rtotal = t1 - t0

    error = risk(mnist.validation[1], y_v)
    erisk = risk(mnist.training[1], y_t)

    return {"error": error, "risk": erisk, "training time": ttotal, "validation time": vtotal, "risk time": rtotal,
            "training size": mnist.training[1].shape[0], "validation size": mnist.validation[1].shape[0]}

if __name__ == "__main__":
    usage = 0.1
    ratio = 0.5
    s0, s1 = 0,4
    mnist, names = load_mnist_data(s0, s1, ratio, usage)

