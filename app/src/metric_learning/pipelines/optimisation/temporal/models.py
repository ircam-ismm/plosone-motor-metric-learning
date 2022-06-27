import logging

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import sklearn.preprocessing as skprep
import scipy.optimize as sopt

import fastdtw

import mlflow

from metric_learning.extras import utils


def ponderate(arr, weights):
    """Create an array of shape arr.shape[0] with weights.shape[0] equi-spaced
    segments of values weights.
    """
    n_weights = weights.shape[0]
    weights = weights.reshape(-1,1)
    region_size = arr.shape[0] // n_weights
    remainder = arr.shape[0] % n_weights
    vec = np.hstack([(weights * np.ones((n_weights, region_size))).reshape(-1),
                     weights[-1] * np.ones(remainder)])
    return vec


from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
def sigmoid_1(x, a, b):
    return a * 1/(1 + np.exp(-b * x)) - 1


class LR(LinearRegression):

    def rmse(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        res = ((y_true - y_pred) ** 2).sum()
        return res / X.shape[0]

class Sigmoid():
    popt = None

    def _sigmoid(self, x, a, b):
        return a * 1/(1 + np.exp(-b * x)) - 1

    def fit(self, X, y):
        popt, pcov = curve_fit(self._sigmoid, X.reshape(-1), y.reshape(-1))
        self.popt = popt
        return self

    def predict(self, X):
        if self.popt is None:
            raise("Fit model before predict.")
        return self._sigmoid(X, *self.popt)

    def rmse(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        res = ((y_true - y_pred) ** 2).sum()
        return res / X.shape[0]


class StopOptimizingException(Exception):
    """Raised in scipy minimize to stop the optimisation.
    """
    pass


class Model(BaseEstimator, RegressorMixin):
    """Regression model for between human annotations and DTW path scores.
    This particular version can optmise n_regions segments on the path score.
    The data is stored in one singleton and the compute is shared across process.
    That is X stores the key to access the data.

    Args:
        users ([type]): [description]
        parameters (dict): Parameters for X, Y, Z
    """
    def __init__(self,
        users,
        templates,
        n_regions,
        init_weights,
        parameters) -> None:
        super().__init__()

        self.log = logging.getLogger(__name__)

        self.parameters = parameters
        self.users = users
        self.templates = templates

        self.n_regions = n_regions
        self.init_weights = init_weights
        self.weights = init_weights
        self.dimensions = parameters['data']['dims']
        self.dtw_radius = parameters['dtw']['radius']
        self.dtw_fun = parameters['dtw']['fun']



    def fit(self, X, y):
        self.step = 0
        self.last_cc_valid = -1

        def cb(xk):
            cc_valid = self.score(X, y, theta=xk)
            self.log.info(xk)
            self.log.info("CV[{}]: {:.3f} {:.3f}".format(self.step, self.last_cc_valid, cc_valid))

            mlflow.log_metric(key="score", value=cc_valid, step=self.step)

            # store the latest weights
            self.weights = xk
            self.step += 1

        X_init = self.init_weights
        n_iter = self.parameters['opt']['n_iter']
        optim_options = {'disp': None, 'gtol': 1e-36, 'eps': 1e-3, 'maxiter': n_iter, }
        optim_bounds = [(1e-5*f,1e5*f) for f in np.ones(self.n_regions)]

        def loss(xk, X, y):
            loss = np.abs(1 - self.score(X, y, xk))
            return loss

        try:
            res = sopt.minimize(loss, X_init, args=(X, y, ),
                                method='L-BFGS-B', callback=cb,
                                options=optim_options,
                                bounds=optim_bounds)

            print("SCIPY RES:", res)
        except StopOptimizingException:
            pass



    def fit_with_validation(self, Xtrain, ytrain, Xvalid, yvalid):
        """This fits the modedtw_l until the validation performance disminishes."""

        self.step = 0
        self.last_cc_valid = -1

        cc_train = self.score(Xtrain, ytrain)

        def cb(xk):
            cc_valid = self.score(Xtrain, ytrain, theta=xk)
            # cc_valid = self.score(Xvalid, yvalid, theta=xk)
            self.log.info("CV[{}]: {:.3f} {:.3f}".format(self.step, self.last_cc_valid, cc_valid))

            # early stopping condition
            if np.isclose(self.last_cc_valid, cc_valid, atol=1e-03):
                self.log.info("EARLY STOPPING: {}".format(self.step))
                raise StopOptimizingException()
            # store the latest weights
            else:
                self.last_cc_valid = cc_valid
                self.weights = xk

            self.step += 1

        X_init = np.ones(self.n_regions)
        n_iter = self.parameters['opt']['n_iter']
        optim_options = {'disp': None, 'gtol': 1e-36, 'eps': 1e-3, 'maxiter': n_iter, }
        optim_bounds = [(1e-5*f,1e5*f) for f in np.ones(self.n_regions)]

        def loss(xk, X, y):
            loss = self.score(X, y, xk)
            reg = np.linalg.norm(1 - self.weights) + 1e-2# / self.weights.shape[0]
            return loss + reg

        try:
            res = sopt.minimize(loss, X_init, args=(Xtrain, ytrain, ),
                                method='L-BFGS-B', callback=cb,
                                options=optim_options,
                                bounds=optim_bounds)
        except StopOptimizingException:
            pass


    def compute(self, X, theta=None) -> float:
        """Score X for target y, with model saved weights or given input weights
        theta."""
        if theta is None:
            theta = self.weights

        # compute
        res = X.apply(self._compute_dtw_row, args=(theta, ), axis=1)
        return res


    def score(self, X, y, theta=None, fun='rmse') -> float:
        """Score X for target y, with model saved weights or given input weights
        theta."""

        res = self.compute(X, theta)

        if fun =='rmse':
            X = np.array(res).reshape(-1,1)
            y = y.values.reshape(-1,1)
            self.lr = LR().fit(X, y)
            score = self.lr.rmse(X, y)

        if fun == 'correlation':
            corrcoef = np.corrcoef(y.values, np.array(res))[0,1]
            score = corrcoef

        return score


    def _compute_dtw_row(self, row, weights):
        """Compute the DTW between both observations described in the row and
        the template. Return the difference between the DTW.
        """
        # select data and normalise
        a, _, path_a, t = self._cached_compute(row['user'], row['day_0'], row['rep_0'])
        b, _, path_b, t = self._cached_compute(row['user'], row['day_1'], row['rep_1'])

        # ponderate
        vec = ponderate(t, np.arange(weights.shape[0]))

        da_ = np.linalg.norm( (a[path_a[:,0]] - t[path_a[:,1]]) , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec[path_a[:, 1]], da_]).T, columns=['s', 'd']).astype({"s":int})
        da_ = (df.groupby("s").agg(self.dtw_fun)['d'] * weights).sum()

        db_ = np.linalg.norm( (b[path_b[:,0]] - t[path_b[:,1]]) , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec[path_b[:, 1]], db_]).T, columns=['s', 'd']).astype({"s":int})
        db_ = (df.groupby("s").agg(self.dtw_fun)['d'] * weights).sum()

        return da_ - db_


    @utils.memoized_method(maxsize=1024)
    def _cached_compute(self, user, day, trial):
        """Cached compute of the DTW between a motion and a template.
        """
        a = utils.select(self.users, gesture=1, user=user, day=day, trial=trial)
        a = skprep.StandardScaler().fit_transform(a[list(self.dimensions)])

        t = utils.select(self.templates, template=1, version=0)
        t = skprep.StandardScaler().fit_transform(t[list(self.dimensions)].iloc[100:])

        da, path_a = fastdtw.fastdtw(a, t, radius=self.dtw_radius)
        path_a = np.array(path_a)
        return a, da, path_a, t


# def main(users, templates, annotations, parameters):

#     if parameters["cv"]['active']:
#         A, B = main_with_cv(users, templates, annotations, parameters)

#     if not parameters["cv"]['active']:
#         A, B = main_no_cv(users, templates, annotations, parameters)

#     return A, B
