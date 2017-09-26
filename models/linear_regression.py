import numpy as np
from optimizers import gradient_descent


class Model(object):

    def __init__(self):
        self.training_data = None
        self.hypothesis_matrix = None
        self.hypothesis_pdf_matrix = None
        self.difference_matrix = None
        self.difference_pdf_matrix = None
        self.cost_matrix = None
        self.num_samples = None
        self.num_features = None
        self.features = None
        self.labels = None
        self.loss = None
        self.optimizer = None

    def train(self, features, labels):
        self.num_samples = len(features)
        self.num_features = len(features[0])

        # theta0 + theta1*X_1 + thera1*X_2 ...., below const is for theta0
        const = np.full((1, self.num_samples), 1)
        print '[Task Definition] This training contains [%s] samples and [%s] features' % (self.num_samples, self.num_features)
        self.features = np.concatenate((np.array(features, dtype=float), const.T), axis=1)
        self.labels = np.array(labels, dtype=float)

        self.optimizer = gradient_descent.LinearRegressionOptimizer(num_samples=self.num_samples, num_features=self.num_features, features=self.features, labels=self.labels)
        self.optimizer.optimize()

        self.hypothesis_matrix = self.optimizer.get_hypothesis()
        self.loss = self.optimizer.get_loss()

    def predict(self, feature):
        _feature = np.array([1] + feature)
        prediction = np.matmul(_feature, self.hypothesis_matrix)
        return prediction

    def get_loss(self):
        return self.loss
