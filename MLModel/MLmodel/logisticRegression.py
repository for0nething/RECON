import numpy as np
from sklearn import metrics

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
    else:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))

    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T


class LogisticRegression(object):
    def __init__(self, dim, num_class):
        self.binary = num_class == 1
        self.W = np.zeros((dim, num_class))
        self.b = np.zeros(num_class)
        self.params = np.array([self.W, self.b])

    def activation(self, input, params=None):
        W, b = params if params is not None else self.params
        if self.binary:
            return sigmoid(np.dot(input, W) + b)
        else:
            return softmax(np.dot(input, W) + b)

    def loss(self, input, label, l2_reg=0.00, params=None):
        sigmoid_activation = self.activation(input, params)

        cross_entropy = - np.mean(np.sum(label * np.log(sigmoid_activation) +
                                         (1 - label) * np.log(1 - sigmoid_activation), axis=1))

        return cross_entropy + l2_reg * np.linalg.norm(self.W) ** 2 / 2

    def f1(self, input, label, params=None):
        if self.binary:
            return metrics.f1_score(label, np.rint(self.predict(input, params)), average = 'weighted')
        else:
            return metrics.f1_score(np.argmax(label, axis=1), np.argmax(self.predict(input, params), axis=1),
                                    average='weighted')
    def recall(self, input, label, params=None):
        if self.binary:
            return metrics.recall_score(label, np.rint(self.predict(input, params)), average = 'weighted')
        else:
            return metrics.recall_score(np.argmax(label,axis=1), np.argmax(np.rint(self.predict(input, params)), axis=1), average = 'weighted')
    def precision(self, input, label, params=None):
        if self.binary:
            return metrics.precision_score(label, np.rint(self.predict(input, params)), average = 'weighted')
        else:
            return metrics.precision_score(np.argmax(label, axis=1), np.argmax(self.predict(input, params),axis=1), average = 'weighted')

    def acc(self, input, label, params=None):
        if self.binary:
            return metrics.accuracy_score(label, np.rint(self.predict(input, params)))
        else:
            if len(label.shape)>1:

                label = np.argmax(label, axis=1)
            pred = self.predict(input, params)
            if len(pred.shape)>1:
                pred = np.argmax(pred, axis=1)
            return metrics.accuracy_score(label,pred)


    def predict(self, input, params=None):
        return self.activation(input, params)

    def accuracy(self, input, label, params=None):
        if self.binary:
            return np.mean(np.isclose(np.rint(self.predict(input, params)), label))
        else:
            if len(label.shape)>1:
                label = np.argmax(label, axis=1)
            pred = self.predict(input, params)
            if len(pred.shape)>1:
                pred = np.argmax(pred, axis=1)
            return metrics.accuracy_score(label,
                                          pred)
    def gradient(self, input, label, l2_reg=0.00, params=None,cnt=1):
        p_y_given_x = self.activation(input, params)
        d_y = label - p_y_given_x
        d_W = -np.dot(np.reshape(input, (cnt, -1)).T, np.reshape(d_y.T, (cnt, -1))) - l2_reg * self.W
        d_b = -np.mean(d_y, axis=0)
        return np.array([d_W, d_b])

    def gradientVec(self, input, label, cnt, l2_reg=0.00, params=None):
        p_y_given_x = self.activation(input, params)
        d_y = label - p_y_given_x
        d_W = -np.dot(np.reshape(input, (cnt, -1)).T, np.reshape(d_y.T, (cnt, -1))) - l2_reg * self.W
        d_b = -np.mean(d_y, axis=0)
        return np.array([d_W, d_b])

    def MASLE(self, X,y):
        predict_y = self.activation(X)
        if len(predict_y.shape)>0:
            predict_y = np.argmax(predict_y, axis=1)

        if len(y.shape) > 0:
            y = np.argmax(y, axis=1)
        MAE = metrics.mean_absolute_error(y, predict_y)
        MSE = metrics.mean_squared_error(y,  predict_y)
        if np.any(y<0):
            MSLE=0
        else:
            MSLE = metrics.mean_squared_log_error(y, predict_y)

        return MAE, MSE, MSLE