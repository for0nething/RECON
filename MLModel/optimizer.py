import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from MLModel.Global import *
class Optimizer(object):

    @staticmethod
    def order_elements(shuffle, n, seed=1234):
        if shuffle == 0:
            indices = np.arange(n)
        elif shuffle == 1:
            indices = np.random.permutation(n)
        elif shuffle == 2:
            indices = np.random.randint(0, n, n)
        else:  # fixed permutation
            np.random.seed(seed)
            indices = np.random.permutation(n)
        return indices

    def optimize(self, method, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        if method == 'sgd':
            return self.sgd(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'saga':
            return self.saga(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'svrg':
            return self.svrg(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method =='BGD':
            return self.BGD(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        else:
            print('Optimizer is not defined!')

    def sgd(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()
        writer = SummaryWriter(CSPATH+'/tensorboard/')
        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]

                model.params -= lr[epoch] * grads
            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)

            writer.add_scalar('loss', model.loss(data,labels), global_step=epoch)
        return W, T

    def BGD(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            # grads_ = None

            grads_ = model.gradient(data, labels,l2_reg, cnt=n)/n
            # print('grads_ is ', grads_)
            # for i in indices:
            #     if grads_ is None:
            #         grads_ = model.gradient(data[i], labels[i],  l2_reg / n) * weights[i]
            #         # grads_ = np.dot(model.gradientVec(data, labels, n, l2_reg / n) , weights)
            #     else:
            #         grads_ += model.gradient(data[i], labels[i],  l2_reg / n) * weights[i]
            #         # grads_ += np.dot(model.gradient(data, labels, n, l2_reg / n) , weights)
            model.params -= lr[epoch] * grads_
            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def saga(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        saved_grads = np.array([model.gradient(data[i], labels[i], l2_reg / n) * weights[i] for i in range(n)])
        avg_saved_grads = saved_grads.mean(axis=0)

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]
                model.params -= lr[epoch] * (grads - saved_grads[i] + avg_saved_grads)
                avg_saved_grads += (grads - saved_grads[i]) / n
                saved_grads[i] = grads

            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def svrg(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            init_grads = np.array([model.gradient(data[i], labels[i], l2_reg / n) * weights[i] for i in range(n)])
            avg_init_grads = np.mean(init_grads, axis=0)

            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]
                model.params -= lr[epoch] * (grads - init_grads[i] + avg_init_grads)

            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T
