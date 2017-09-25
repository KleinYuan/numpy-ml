from __future__ import absolute_import
import unittest
from models import linear_regression


class TestLinearRegressionModel(unittest.TestCase):

    def setUp(self):
        self.features = [[1, 1.2], [2, 2.2], [3, 3.2], [4, 4.3], [5, 5.2], [6, 6.1], [7, 7.5]]
        self.labels = [12, 23, 36, 45, 51, 67, 79]
        self.test_feature = [2.1, 3.5]
        self.pass_test_loss = 2
        self.model = linear_regression.Model()
        self.model.train(features=self.features, labels=self.labels)

    def test_loss(self):
        loss = self.model.get_loss()
        self.assertTrue(loss <= self.pass_test_loss)

    def test_prediction(self):
        prediction = self.model.predict(feature=self.test_feature)
        self.assertTrue(20 <= prediction <= 25)

if __name__ == '__main__':
    unittest.main()
