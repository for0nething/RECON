import numpy as np
from sklearn import metrics
class LinearRegression(object):
    def __init__(self, dim):
        self.W = np.zeros(dim)
        self.params = self.W

    def activation(self, X, params=None):
        pred_ys = X.dot(self.W)
        return pred_ys

    def loss(self, X,y, l2_reg=0.00, ):
        num_of_samples = X.shape[0]
        f_mat = X.dot(self.W)
        diff = f_mat - y
        loss = 1.0 * np.sum(diff * diff) / num_of_samples

        return loss + l2_reg * np.linalg.norm(self.W) ** 2 / 2

    def gradient(self, X, y, l2_reg=0.00, params=None, cnt=0):
        num_of_samples = X.shape[0]
        f_mat = X.dot(self.W)
        diff = f_mat - y
        if type(diff)==np.array and diff.shape[0]==1:
            gradient = (diff[0]*(X)).T - l2_reg * self.W
            return gradient
        else:
            if type(diff) ==np.float64:
                gradient = (diff *X).T - l2_reg * self.W
            else:
                gradient = ((diff.T).dot(X)).T - l2_reg * self.W
            return gradient


    def MASLE(self, X,y):
        predict_y = self.activation(X)
        MAE = metrics.mean_absolute_error(y, predict_y)
        MSE = metrics.mean_squared_error(y,  predict_y)
        MSLE=0
        return MAE, MSE, MSLE