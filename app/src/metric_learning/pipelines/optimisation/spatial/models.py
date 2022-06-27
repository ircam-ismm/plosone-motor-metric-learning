import logging
import multiprocessing as mp

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
import scipy.optimize as sopt

from metric_learning.extras import utils
from . import compute

from sklearn.linear_model import LinearRegression
class LR(LinearRegression):

    def rmse(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        res = ((y_true - y_pred) ** 2).sum()
        return res / X.shape[0]

    def mae(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        res = np.linalg.norm(y_true - y_pred, ord=1)
        return res / X.shape[0]


class PackedIterator:
    """Packs an iterator with some data. For each iteration, a tuple is returned
    with a new element of the iterator and the data in first and second position,
    respectively.
    """
    def __init__(self, iterator, data):
        self.iterator = iterator
        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        return (next(self.iterator), self.data)


class StopOptimizingException(Exception):
    """Raised in scipy minimize to stop the optimisation.
    """
    pass



class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self, parameters):

        super().__init__()
        self.log = logging.getLogger(__name__)
        self.parameters = parameters

        # opt
        self.eps = parameters['opt']['eps']
        self.init_scale = parameters['opt']['init_scale']
        self.C = parameters['opt']['regularisation']

        # dtw
        self.dimensions = parameters['data']['dims']
        self.radius = parameters['dtw']['radius']

        # data
        self.data_shape = len(parameters['data']['dims'])
        self.distance = parameters['dtw']['distance']


    def fit(self, X, y):
        self.step = 0

        def cb(xk):
            score = self.score(X, y, theta=xk)

            # A = np.eye(self.data_shape) * xk[:self.data_shape]
            # B = xk[self.data_shape:].reshape(self.data_shape, self.data_shape)
            # M = B@A@B.T
            # reg = np.linalg.norm(np.eye(self.data_shape) - M)

            self.log.info(xk)
            self.log.info("CV[{}]: {:.3f}".format(self.step, score))

            # store the latest weights
            self.weights = xk
            self.step += 1

        X_init = self.init_weights
        n_iter = self.parameters['opt']['n_iter']
        optim_options = {'disp': None, 'maxls': 50, 'iprint': -1,
                         'gtol': 1e-8, 'ftol': 1e-8, 'eps': self.eps,
                         'maxiter': n_iter}

        A = [(1e-5*f,1e5*f) for f in np.ones(self.data_shape)]
        if self.distance == 'full':
            B = [(-1e5*f,1e5*f) for f in np.ones(self.weights_shape)]
        else:
            B = []
        optim_bounds = A + B

        try:
            res = sopt.minimize(self.loss, X_init, args=(X, y, ),
                                method='L-BFGS-B', callback=cb,
                                options=optim_options,
                                bounds=optim_bounds)

            print("SCIPY RES:", res)
        except StopOptimizingException:
            pass


    def compute(self, X, theta=None):
        """Score X for target y, with model saved weights or given input weights
        theta."""
        if theta is None:
            theta = self.weights

        # compute
        res = self.pool.map("compute", PackedIterator(X.iterrows(), (theta, self.dimensions)))
        return res


    def score(self, X, y, theta=None, fun='rmse'):
        """Score X for target y, with model saved weights or given input weights
        theta."""

        res = self.compute(X, theta)

        if fun == 'proj':
            Xmax = np.abs(res).max()
            score = ((y.values - res / Xmax)**2).mean()

        if fun =='rmse':
            X = np.array(res).reshape(-1,1)
            y = y.values.reshape(-1,1)
            self.lr = LR().fit(X, y)
            score = self.lr.rmse(X, y)

        if fun == 'correlation':
            corrcoef = np.corrcoef(y.values, np.array(res))[0,1]
            score = np.abs(1 - corrcoef)

        return score


class DiagonalCovariance(BaseModel):
    def __init__(self, parameters):

        super().__init__(parameters)

        self.weights_shape = self.data_shape
        self.init_weights = np.ones(self.weights_shape) * self.init_scale
        self.weights = self.init_weights

        # update the class variables for myprocess
        compute.MyProcess.distance = self.distance
        compute.MyProcess.radius = self.radius
        compute.MyProcess.weights_shape = self.weights_shape
        compute.MyProcess.data_shape = self.data_shape

        ctx = mp.get_context()  # get the default context
        ctx.Process = compute.MyProcess  # override the context's Process
        self.pool = ctx.Pool(parameters['opt']['n_cores'])


    def loss(self, xk, X, y):
        loss = self.score(X, y, xk, fun='rmse')
        reg = np.linalg.norm(self.weights) / self.weights.shape[0]
        return loss + self.C * reg


class FullCovariance(BaseModel):
    def __init__(self, parameters):

        super().__init__(parameters)

        self.weights_shape = self.data_shape * self.data_shape
        self.init_weights = np.hstack([np.ones(self.data_shape), np.eye(self.data_shape).reshape(-1)])
        self.weights = self.init_weights

        # processing pool
        compute.MyProcess.distance = self.distance
        compute.MyProcess.radius = self.radius
        compute.MyProcess.weights_shape = self.weights_shape
        compute.MyProcess.data_shape = self.data_shape

        ctx = mp.get_context()  # get the default context
        ctx.Process = compute.MyProcess  # override the context's Process
        self.pool = ctx.Pool(parameters['opt']['n_cores'])


    def loss(self, xk, X, y):
        loss = self.score(X, y, xk, fun='rmse')

        A = np.eye(self.data_shape) * xk[:self.data_shape]
        B = xk[self.data_shape:].reshape(self.data_shape, self.data_shape)
        M = B@A@B.T

        # reg = np.linalg.norm(np.eye(self.data_shape) - M)
        # w_diag = self.weights[:self.data_shape]
        # reg_diag = np.linalg.norm(w_diag)
        # w_full = self.weights[self.data_shape:].reshape(self.data_shape, self.data_shape)
        # reg_full = np.linalg.norm(np.eye(self.data_shape) - w_full)
        # reg = reg_diag + reg_full

        return loss #+ self.C * reg
