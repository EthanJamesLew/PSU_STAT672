''' Label Data
@author: Ethan Lew
(elew@pdx.edu)

Objectifies the notion of data and labeled data.
'''
import numpy as np


class Data:
    def __init__(self):
        self._X = None

    @property
    def n(self):
        if self._X is None:
            return None
        else:
            return np.shape(self._X)[0]

    @property
    def d(self):
        if self._X is None:
            return None
        else:
            return np.shape(self._X)[1]

    @property
    def x(self):
        return self._X

    def add_data(self, x):
        if self._X is None:
            self._X = x
        else:
            # Dimensionality Check
            if self.n != np.shape(x)[0]:
                raise Exception("New X data doesn't match dimensionality to the old data! (%s, %s)" %
                                (self.n, np.shape(x)[0]))
            self._X = np.vstack((self._X, x))

    def __getitem__(self, i):
        return self._X[i]


class LabelData(Data):
    def __init__(self):
        self._Y = None
        super(LabelData, self).__init__()

    @property
    def y(self):
        return self._Y

    def add_data(self, x, y):
        # Add observations
        if self._X is None:
            self._X = x
            self._Y = y
        else:
            # Dimensionality Check
            if self.n != np.shape(x)[0]:
                raise Exception("New X data doesn't match dimensionality to the old data! (%s, %s)" %
                                (self.n, np.shape(x)[0]))
            self._X = np.vstack((self._X, x))
            self._Y = np.vstack((self._Y, y))

    def __getitem__(self, i):
        return self._X[i], self._Y[i]

class PartitionData():
    def __init__(self, data):
        super(PartitionData, self).__init__()
        self._data = data
        self._datat = data
        self._datav = data

    def partition(self, ratio):
        M = round(self._data.n * ratio)
        train = np.zeros((self._data.n), dtype=np.bool)
        train[0:M] = 1
        np.random.shuffle(train)
        self._datat = self._data[train]
        self._datav = self._data[~train]

    @property
    def training(self):
        return self._datat

    @property
    def validation(self):
        return self._datav

    def __getitem__(self, i):
        return self._data[i]

if __name__ == "__main__":
    data = np.array([[2,3],[1,2],[7,8],[9,3]])
    labs = np.array([0.4, 1.2, 3.4, -0.4])
    d = LabelData()
    d.add_data(data, labs)
    p = PartitionData(d)
    p.partition(0.2)
    print(p.validation)
