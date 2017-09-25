import numpy as np

'''
PDF below means Partial Differentiation Function
'''


class LinearRegression(object):

    def __init__(self, converge_loss=2):
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
        self.converge_loss = converge_loss
        self.epoch_cnt = 0

    def train(self, features, labels):
        self.num_samples = len(features)
        self.num_features = len(features[0])

        # theta0 + theta1*X_1 + thera1*X_2 ...., below const is for theta0
        const = np.full((1, self.num_samples), 1)
        print '[Task Definition] This training contains [%s] samples and [%s] features' % (self.num_samples, self.num_features)
        self.features = np.concatenate((np.array(features, dtype=float), const.T), axis=1)
        self.labels = np.array(labels, dtype=float)
        self._init_params()

        # # TODO: We need to separate optimizer out as a seprate moduel but for now we keep it simpler here
        learning_rate = 0.001
        while self.loss >= self.converge_loss:
            self.difference_pdf_matrix = np.matmul(self.difference_matrix, self.hypothesis_pdf_matrix) / self.num_samples
            self.hypothesis_matrix = self.hypothesis_matrix - learning_rate * self.difference_pdf_matrix
            self._update_params()
            self.epoch_cnt += 1
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

    def predict(self, feature):
        _feature = np.array([1] + feature)
        prediction = np.matmul(_feature, self.hypothesis_matrix)
        return prediction


def test():
    features = [[1, 1.2], [2, 2.2], [3, 3.2], [4, 4.3], [5, 5.2], [6, 6.1], [7, 7.5]]
    labels = [12, 23, 36, 45, 51, 67, 79]
    model = LinearRegression()
    model.train(features=features, labels=labels)

    test_feature = [2.1, 3.5]
    prediction = model.predict(feature=test_feature)
    print '[Prediction] %s' % prediction

if __name__ == '__main__':
    test()
