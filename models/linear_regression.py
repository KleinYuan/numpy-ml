import numpy as np


class LinearRegression(object):

    def __init__(self, converge_loss=0.005):
        self.training_data = None
        self.hypothesis_matrix = None
        self.loss_matrix = None
        self.num_samples = None
        self.num_features = None
        self.features = None
        self.labels = None
        self.loss = None
        self.optimizer = None
        self.converge_loss = converge_loss

    def train(self, features, labels, optimizer):
        self.optimizer = optimizer
        self.num_samples = len(features)
        self.num_features = len(features[0])
        print '[Task Definition] This training contains [%s] samples and [%s] features' % (self.num_samples, self.num_features)
        self.features = np.array(features, dtype=float)
        self.labels = np.array(labels, dtype=float)

        self._init_params()

    def _init_params(self):
        init_params = np.random.rand(self.num_features)
        self.hypothesis_matrix = init_params
        self.loss_matrix = np.square(np.matmul(self.features, self.hypothesis_matrix) - self.labels)
        self.loss = np.mean(self.loss_matrix)
        print '[Init Loss] %s' % self.loss


def test():
    features = [[1, 1.2], [2, 2.2], [3, 3.2], [4, 4.3], [5, 5.2], [6, 6.1], [7, 7.5]]
    labels = [12, 23, 36, 45, 51, 67, 79]
    model = LinearRegression()
    model.train(features=features, labels=labels)


if __name__ == '__main__':
    test()
