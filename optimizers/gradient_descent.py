import numpy as np


class LinearRegressionOptimizer(object):
    def __init__(self, num_samples, num_features, features, labels, converge_loss=2):
        self.converge_loss = converge_loss
        self.features = features
        self.labels = labels
        self.num_samples = num_samples
        self.num_features = num_features
        self.hypothesis_matrix = None
        self.loss = None
        self.difference_matrix = None
        self.difference_pdf_matrix = None
        self.cost_matrix = None
        self.epoch_cnt = 0
        self._init_params()

    def optimize(self):
        learning_rate = 0.001
        while self.loss >= self.converge_loss:
            self.difference_matrix = np.matmul(self.features, self.hypothesis_matrix) - self.labels
            self.difference_pdf_matrix = np.matmul(self.difference_matrix, self.hypothesis_pdf_matrix) / self.num_samples
            self.hypothesis_matrix = self.hypothesis_matrix - learning_rate * self.difference_pdf_matrix
            self._update_params()
            self.epoch_cnt += 1
            if self.epoch_cnt % 500 == 0:
                print '[%sth Loss] %s' % (self.epoch_cnt, self.loss)

    def _init_params(self):
        init_params = np.random.rand(self.num_features + 1)
        self.hypothesis_matrix = init_params
        self._update_params()
        print '[Init Loss] %s' % self.loss

    def _update_params(self):
        theta_0 = self.hypothesis_matrix[0]
        theta_0_vector = np.full((1, self.num_samples), theta_0)
        self.hypothesis_pdf_matrix = np.concatenate((theta_0_vector.T, self.features[:, 1:]), axis=1)
        self.difference_matrix = np.matmul(self.features, self.hypothesis_matrix) - self.labels
        self.cost_matrix = np.square(self.difference_matrix) * 0.5
        self.loss = np.mean(self.cost_matrix)

    def get_hypothesis(self):
        return self.hypothesis_matrix

    def get_loss(self):
        return self.loss