import logging
import multiprocessing as mp

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RepeatedKFold
import sklearn.model_selection as skms
import sklearn.preprocessing as skprep

import scipy.optimize as sopt

import mlflow
from mlflow.utils.file_utils import TempDir

import pingouin as pg

from metric_learning.extras import utils
from . import myprocess

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



weights_ind = np.array([ 0.91392964,  0.99812938,  0.38325155,  1.09571258,  0.79930309,
        1.41883998,  1.20781645,  1.06462606,  1.01527823,  0.8005372 ,
        0.14734859, -0.21480207,  0.04144787,  0.19062913,  0.2749446 ,
        0.27830216, -0.20713299, -0.01499813,  0.14612732,  0.98392723,
       -0.08390575,  0.23760452,  0.17372372, -0.03425151,  0.14724068,
        0.13352981,  0.11515776, -0.23428854, -0.09390888, -0.41620132,
        0.12604057, -0.22772833, -0.11193977, -0.03145384, -0.2805427 ,
       -0.01991797,  0.04055951,  0.23791334,  0.11138086,  1.15958291,
       -0.16114595,  0.31548395, -0.09428427, -0.08316355,  0.23170692,
        0.19189682,  0.17901146, -0.20579617, -0.16750402,  0.56339851,
        0.08608745,  0.1160275 , -0.12514889,  0.16717858,  0.26684696,
       -0.03337617, -0.09167362,  0.3072943 ,  0.08292372,  1.78748918,
        0.2222252 , -0.00266819, -0.03684664,  0.26857366,  0.14504965,
       -0.032821  , -0.09272178,  0.11300847,  0.2234396 ,  1.37938417,
        0.01656942, -0.12865516, -0.20558658,  0.13409503, -0.25063482,
       -0.08362311, -0.12179413, -0.00207164,  0.01691972,  1.10768758,
        0.1492121 , -0.0148081 ,  0.11449339, -0.0188327 ,  0.23034026,
        0.16256516, -0.03744404, -0.1316143 ,  0.14722634,  1.01504774])

class Model(BaseEstimator, RegressorMixin):
    def __init__(self, parameters) -> None:

        super().__init__()
        self.log = logging.getLogger(__name__)
        self.parameters = parameters

        self.data_shape = len(parameters['data']['dims'])

        self.distance = parameters['dtw']['distance']
        n = self.data_shape
        init_scale = parameters['opt']['init_scale']

        if self.distance == 'diag':
            weights_shape = n
            self.init_weights = np.ones(weights_shape) * init_scale

        if self.distance == 'full':
            weights_shape = n**2
            self.init_weights = np.hstack([np.ones(self.data_shape), np.eye(self.data_shape).reshape(-1)])
            self.init_weights = weights_ind

        self.weights_shape = weights_shape
        self.weights = self.init_weights
        # dtw
        self.dimensions = parameters['data']['dims']
        self.radius = parameters['dtw']['radius']
        # opt
        self.eps = parameters['opt']['eps']


        # update the class variables for myprocess
        myprocess.MyProcess.distance = self.distance
        myprocess.MyProcess.radius = self.radius
        myprocess.MyProcess.weights_shape = self.weights_shape
        myprocess.MyProcess.data_shape = self.data_shape

        # processing pool
        ctx = mp.get_context()  # get the default context
        ctx.Process = myprocess.MyProcess  # override the context's Process
        self.pool = ctx.Pool(parameters['opt']['n_cores'])


    def fit(self, X, y):
        self.step = 0
        self.last_cc_valid = -1

        def cb(xk):
            cc_valid = self.score(X, y, theta=xk)
            self.log.info(xk)
            self.log.info("CV[{}]: {:.3f} {:.3f}".format(self.step, self.last_cc_valid, cc_valid))

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

        def loss(xk, X, y):
            loss = self.score(X, y, xk, fun='rmse')

            if self.distance == 'diag':
                reg = np.linalg.norm(self.weights) / self.weights.shape[0]

            if self.distance == 'full':
                w_diag = self.weights[:self.data_shape]
                reg_diag = np.linalg.norm(w_diag) / w_diag.shape[0]
                w_full = self.weights[self.data_shape:].reshape(self.data_shape, self.data_shape)
                reg_full = np.linalg.norm(np.eye(self.data_shape) - w_full) / self.data_shape**2
                reg = reg_diag + reg_full

            return loss + reg

        try:
            res = sopt.minimize(loss, X_init, args=(X, y, ),
                                method='L-BFGS-B', callback=cb,
                                options=optim_options,
                                bounds=optim_bounds)

            print("SCIPY RES:", res)
        except StopOptimizingException:
            pass


    def fit_with_validation(self, Xtrain, ytrain, Xvalid, yvalid):
        """This fits the model until the validation performance disminishes."""

        self.step = 0
        self.last_cc_valid = -1

        def cb(xk):
            cc_valid = self.score(Xtrain, ytrain, theta=xk, fun='correlation')
            self.log.info(xk)
            self.log.info("CV[{}]: {:.3f} {:.3f}".format(self.step, self.last_cc_valid, cc_valid))

            # early stopping condition
            # if (self.last_cc_valid > cc_valid):
            if np.isclose(self.last_cc_valid, cc_valid, atol=1e-03):
                self.log.debug("EARLY STOPPING: {}".format(self.step))
                raise StopOptimizingException()
            # store the latest weights
            else:
                self.last_cc_valid = cc_valid
                self.weights = xk

            self.step += 1

        X_init = self.init_weights
        n_iter = self.parameters['opt']['n_iter']

        # optim_options = {'disp': None, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': n_iter, }
        optim_options = {'disp': None, 'maxls': 50, 'iprint': -1,
                 'gtol': 1e-8, 'ftol': 1e-8, 'eps': self.eps,
                 'maxiter': n_iter}

        A = [(1e-5*f,1e5*f) for f in np.ones(self.data_shape)]
        if self.distance == 'full':
            B = [(-1e5*f,1e5*f) for f in np.ones(self.weights_shape)]
        else:
            B = []
        optim_bounds = A + B

        def loss(xk, X, y):
            loss = self.score(X, y, xk, fun='rmse')

            if self.distance == 'diag':
                reg = np.linalg.norm(self.weights) / self.weights.shape[0]

            if self.distance == 'full':
                w_diag = self.weights[:self.data_shape]
                reg_diag = np.linalg.norm(1 - w_diag) / w_diag.shape[0]
                w_full = self.weights[self.data_shape:].reshape(self.data_shape, self.data_shape)
                reg_full = np.linalg.norm(np.eye(self.data_shape) - w_full) / self.data_shape**2
                reg = reg_diag + reg_full + 1e-2

            return loss + reg

        try:
            res = sopt.minimize(loss, X_init, args=(Xtrain, ytrain, ),
                                method=self.parameters['opt']['method'], callback=cb,
                                options=optim_options,
                                bounds=optim_bounds)

            print("SCIPY RES:", res)
        except StopOptimizingException:
            pass


    def compute(self, X, theta=None) -> float:
        """Score X for target y, with model saved weights or given input weights
        theta."""
        if theta is None:
            theta = self.weights

        # compute
        res = self.pool.map("compute", PackedIterator(X.iterrows(), (theta, self.dimensions)))
        return res


    def score(self, X, y, theta=None, fun='rmse') -> float:
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


# def train_test_split(annotations, train_ids, test_ids):
#     """Implement the same functionality as sklearn, but with our data. This uses
#     the user id in the dataframe to select folds that are split by users.
#     """
#     X = annotations[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']]
#     y = annotations[['user', 'm']]

#     users_ids = np.array(list(set(annotations['user'])))

#     Xtrain = utils.select(X, user=list(users_ids[train_ids]))
#     ytrain = utils.select(y, user=list(users_ids[train_ids]))['m']
#     Xtest = utils.select(X, user=list(users_ids[test_ids]))
#     ytest = utils.select(y, user=list(users_ids[test_ids]))['m']

#     return Xtrain, ytrain, Xtest, ytest

def train_test_split(annotations, train_ids, test_ids):
    """Implement the same functionality as sklearn, but with our data. This uses
    the user id in the dataframe to select folds that are split by users.
    """
    X = annotations[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']]
    y = annotations[['user', 'm']]

    users_ids = np.array(list(set(annotations['user'])))

    Xtrain = utils.select(X, user=list(users_ids[train_ids]))
    ytrain = utils.select(y, user=list(users_ids[train_ids]))['m']
    Xtest = utils.select(X, user=list(users_ids[test_ids]))
    ytest = utils.select(y, user=list(users_ids[test_ids]))['m']

    return Xtrain, ytrain, Xtest, ytest

def main_no_cv(annotations, parameters):
    """This is meant to optimise full on one dataset and test on the other one.
    The two dataset used here are individual and mean.
    """

    distance = parameters['dtw']['distance']

    data_shape = len(parameters['data']['dims'])
    if distance == 'diag': weights_shape = data_shape
    if distance == 'full': weights_shape = data_shape + data_shape**2

    # columns = ['fold_id', 'train_ids', 'test_ids', 'cc_start', 'cc_end']
    # columns += ['w'+str(i) for i in range(weights_shape)]
    # report = pd.DataFrame(columns=columns)

    data_ind, data_mean = annotations.iloc[90:], annotations.iloc[:90]
    Xmean, ymean = data_mean[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], data_mean['m']
    Xind, yind = data_ind[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], data_ind['m']

    experiment_id = parameters['mlflow']['experiment_id']
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(utils.flatten(parameters))

        # init all parameters
        model = Model(parameters)

        # baseline on mean and ind
        cc_mean_0 = 1-model.score(Xmean, ymean, fun='correlation')
        logging.info("SCORED: {}".format(cc_mean_0))
        mlflow.log_metric(key="cc_mean_0", value=cc_mean_0)

        cc_ind_0 = 1-model.score(Xind, yind, fun='correlation')
        logging.info("SCORED: {}".format(cc_ind_0))
        mlflow.log_metric(key="cc_ind_0", value=cc_ind_0)

        # train the model
        #model.fit(Xmean, ymean)
        model.fit(Xind, yind)

        # final performance on test
        cc_mean_1 = 1-model.score(Xmean, ymean, fun='correlation')
        logging.info("FITTED: {}".format(cc_mean_1))
        mlflow.log_metric(key="cc_mean_1", value=cc_mean_1)

        cc_ind_1 = 1-model.score(Xind, yind, fun='correlation')
        logging.info("FITTED: {}".format(cc_ind_1))
        mlflow.log_metric(key="cc_ind_1", value=cc_ind_1)

        report = pd.DataFrame(data=model.weights)
        with TempDir() as tmp:
            file_path = tmp.path("report.csv")
            with open(file_path, 'w') as f: report.to_csv(f)
            mlflow.log_artifact(file_path)

        with TempDir() as tmp:
            file_path = tmp.path("src.txt")
            import inspect
            import sys
            data = inspect.getsource(sys.modules[__name__])
            with open(file_path, 'w') as f:
                for line in data: f.write(line)
            mlflow.log_artifact(file_path)

    return False, parameters


def main_with_cv(annotations, parameters):
    """This is meant to cross validate many times to get a distribution of the
    weights.
    """
    distance = parameters['dtw']['distance']

    data_shape = len(parameters['data']['dims'])
    if distance == 'diag': weights_shape = data_shape
    if distance == 'full': weights_shape = data_shape + data_shape**2

    columns = ['fold_id', 'train_ids', 'test_ids', 'cc_start', 'cc_end']
    columns += ['w'+str(i) for i in range(weights_shape)]

    report = pd.DataFrame(columns=columns)

    experiment_id = parameters['mlflow']['experiment_id']
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(utils.flatten(parameters))

        # init all parameters
        model = Model(parameters)

        # cross validate the experiment
        n_splits = parameters['cv']['n_splits']
        n_repeats = parameters['cv']['n_repeats']
        random_state = parameters['cv']['random_state']
        test_size = parameters['cv']['test_size']

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        users_ids = np.array(list(set(annotations['user'])))
        # row_ids = np.arange(annotations.shape[0])

        for fold_id, (train_ids, test_ids) in enumerate(rkf.split(users_ids)):
            # reset model weights between folds
            model.weights = model.init_weights

            Xtrain2, ytrain2, Xtest, ytest = train_test_split(annotations, train_ids, test_ids)
            # train2_ids, valid_ids = skms.train_test_split(train_ids, test_size=test_size, random_state=random_state)
            # Xtrain2, ytrain2, Xvalid, yvalid = train_test_split(annotations, train2_ids, valid_ids)
            logging.info("LOOP {}: {}".format(fold_id, (train_ids, True, test_ids)))

            # baseline on test
            cc_start = 1-model.score(Xtest, ytest, fun='correlation')
            logging.info("SCORED: {}".format(cc_start))
            # train the model
            model.fit_with_validation(Xtrain2, ytrain2, True, True)
            # final performance on test
            cc_end = 1-model.score(Xtest, ytest, fun='correlation')
            logging.info("FITTED: {}".format(cc_end))
            # report for current fold
            row = pd.DataFrame(data=[[fold_id, train_ids, test_ids, cc_start, cc_end]+list(model.weights)],
                            columns=columns)
            report = report.append(row)

        # score model on weights' init/average and full dataset
        X, y = annotations[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], annotations['m']
        model.weights = model.init_weights
        cc_init = 1-model.score(X, y, fun='correlation')
        mlflow.log_metric(key="cc_init", value=cc_init)

        weights = report.filter(regex="w.*").mean().values
        model.weights = weights
        cc_all = 1-model.score(X, y, fun='correlation')
        mlflow.log_metric(key="cc_final", value=cc_all)

        # log the relative correlation change
        mlflow.log_metric(key="cc_end", value=report['cc_end'].mean())
        mlflow.log_metric(key="cc_start", value=report['cc_start'].mean())
        diff_rel = (report['cc_end'].mean() - report['cc_start'].mean()) / report['cc_start'].mean()
        mlflow.log_metric(key="diff_rel", value=diff_rel)

        # ttest on final performance change
        ttest = pg.ttest(report['cc_start'], report['cc_end'])
        mlflow.log_metric(key="p-value", value=ttest['p-val'].values[0])
        mlflow.log_metric(key="power", value=ttest['power'].values[0])

        with TempDir() as tmp:
            file_path = tmp.path("report.csv")
            with open(file_path, 'w') as f: report.to_csv(f)
            mlflow.log_artifact(file_path)

        with TempDir() as tmp:
            file_path = tmp.path("params.txt")
            import json
            with open(file_path, 'w') as f: json.dump(parameters, f)
            mlflow.log_artifact(file_path)

        with TempDir() as tmp:
            file_path = tmp.path("src.txt")
            import inspect
            import sys
            data = inspect.getsource(sys.modules[__name__])
            with open(file_path, 'w') as f:
                for line in data: f.write(line)
            mlflow.log_artifact(file_path)

    return report, parameters


from metric_learning.pipelines import select_annotations


def main(annotations, parameters):

    # register log file
    # create file handler which logs even debug messages
    # fh = logging.FileHandler('./run.log', mode='w')
    # fh.setLevel(logging.DEBUG)

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)

    # logger = logging.getLogger('kedro')
    # logger.addHandler(fh)



    if parameters["cv"]["active"]:
        A, B = main_with_cv(annotations, parameters)

    if not parameters["cv"]["active"]:
        A, B = main_no_cv(annotations, parameters)

    return A, B

from kedro.pipeline import Pipeline, node
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=select_annotations,
                inputs="parameters",
                outputs="annotations",
                name="select_annotations_node"
            ),

            node(
                func=main,
                inputs=["annotations", "parameters"],
                outputs=["report", "config"],
                name="spatial_correlation_node"
            ),
        ],
        tags="compute"
    )
