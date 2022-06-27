import logging

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd

import sklearn.model_selection as skms

import mlflow
from mlflow.utils.file_utils import TempDir

import pingouin as pg

from metric_learning.extras import utils

from .models import FullCovariance, DiagonalCovariance


def log_source_code(logname, modulename):
    with TempDir() as tmp:
        file_path = tmp.path(logname)
        import inspect
        data = inspect.getsource(modulename)
        with open(file_path, 'w') as f:
            for line in data: f.write(line)
        mlflow.log_artifact(file_path)


def main_no_cv(annotations, parameters):
    """This is meant to optimise full on one dataset and test on the other one.
    The two dataset used here are individual and mean.
    """
    data_ind, data_mean = annotations.iloc[90:], annotations.iloc[:90]
    Xmean, ymean = data_mean[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], data_mean['m']
    Xind, yind = data_ind[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], data_ind['m']

    experiment_id = parameters['mlflow']['experiment_id']
    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_params(utils.flatten(parameters))

        distance = parameters['dtw']['distance']
        if distance == 'diag': model = DiagonalCovariance(parameters)
        if distance == 'full': model = FullCovariance(parameters)

        # baseline on mean and ind
        cc_mean_0 = 1-model.score(Xmean, ymean, fun='correlation')
        logging.info("baseline mean: {}".format(cc_mean_0))
        mlflow.log_metric(key="cc_mean_0", value=cc_mean_0)

        cc_ind_0 = 1-model.score(Xind, yind, fun='correlation')
        logging.info("baseline ind: {}".format(cc_ind_0))
        mlflow.log_metric(key="cc_ind_0", value=cc_ind_0)

        # train the model
        model.fit(Xmean, ymean)
        # model.fit(Xind, yind)

        # final performance on test
        cc_mean_1 = 1-model.score(Xmean, ymean, fun='correlation')
        logging.info("overfit mean: {}".format(cc_mean_1))
        mlflow.log_metric(key="cc_mean_1", value=cc_mean_1)

        cc_ind_1 = 1-model.score(Xind, yind, fun='correlation')
        logging.info("overfit ind: {}".format(cc_ind_1))
        mlflow.log_metric(key="cc_ind_1", value=cc_ind_1)

        report = pd.DataFrame(data=model.weights)
        with TempDir() as tmp:
            file_path = tmp.path("report.csv")
            with open(file_path, 'w') as f: report.to_csv(f)
            mlflow.log_artifact(file_path)

        import sys
        import metric_learning
        log_source_code("training.py", sys.modules[__name__])
        log_source_code("models.py", metric_learning.pipelines.optimisation.spatial.models)
        log_source_code("compute.py", metric_learning.pipelines.optimisation.spatial.compute)

        with TempDir() as tmp:
            mlflow.log_artifact("./logs/run.log")

    return False, parameters



def train_test_split(annotations, train_ids, test_ids):
    """Implement the same functionality as sklearn, but with our data. This uses
    the user id in the dataframe to select folds that are split by users.
    """
    X = annotations[['user', 'day_0', 'rep_0', 'day_1', 'rep_1', 'dataset']]
    y = annotations[['user', 'm', 'dataset']]

    users_ids = np.array(list(set(annotations['user'])))

    Xtrain = utils.select(X, user=list(users_ids[train_ids]))
    ytrain = utils.select(y, user=list(users_ids[train_ids]))
    Xtest = utils.select(X, user=list(users_ids[test_ids]))
    ytest = utils.select(y, user=list(users_ids[test_ids]))

    return Xtrain, ytrain, Xtest, ytest


def create_report(data_shape, distance):
    if distance == 'diag': weights_shape = data_shape
    if distance == 'full': weights_shape = data_shape + data_shape**2
    columns = ['fold_id', 'train_ids', 'test_ids', 'cc_start', 'cc_end']
    columns += ['w'+str(i) for i in range(weights_shape)]
    report = pd.DataFrame(columns=columns)
    return report


def main_with_cv(annotations, parameters):
    """This is meant to cross validate many times to get a distribution of the
    weights.
    """
    distance = parameters['dtw']['distance']
    data_shape = len(parameters['data']['dims'])
    report = create_report(data_shape, distance)

    experiment_id = parameters['mlflow']['experiment_id']
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(utils.flatten(parameters))

        if distance == 'diag': model = DiagonalCovariance(parameters)
        if distance == 'full': model = FullCovariance(parameters)

        n_splits = parameters['cv']['n_splits']
        n_repeats = parameters['cv']['n_repeats']
        random_state = parameters['cv']['random_state']

        users_ids = np.array(list(set(annotations['user'])))

        rkf = skms.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for fold_id, (train_ids, test_ids) in enumerate(rkf.split(users_ids)):
            # reset model weights between folds
            model.weights = model.init_weights

            Xtrain, ytrain, Xtest, ytest = train_test_split(annotations, train_ids, test_ids)
            logging.info("LOOP {}: {}".format(fold_id, (train_ids, True, test_ids)))

            Xtest_mean = utils.select(Xtest, dataset='mean')
            ytest_mean = utils.select(ytest, dataset='mean')

            # baseline on test
            cc_start = 1-model.score(Xtest_mean, ytest_mean['m'], fun='correlation')
            logging.info("baseline mean: {}".format(cc_start))
            # train the model
            model.fit(Xtrain, ytrain['m'])
            # final performance on test
            cc_end = 1-model.score(Xtest_mean , ytest_mean['m'], fun='correlation')
            logging.info("optimised mean: {}".format(cc_end))
            # report for current fold
            row = pd.DataFrame(data=[[fold_id, train_ids, test_ids, cc_start, cc_end]+list(model.weights)],
                            columns=report.columns)
            report = report.append(row)

        # score model on weights' init/average and full dataset
        X, y = annotations.iloc[:90][['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], annotations.iloc[:90]['m']
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
        ttest = pg.ttest(report['cc_start'], report['cc_end'], paired=True)
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

        import sys
        import metric_learning
        log_source_code("training.py", sys.modules[__name__])
        log_source_code("models.py", metric_learning.pipelines.optimisation.spatial.models)
        log_source_code("compute.py", metric_learning.pipelines.optimisation.spatial.compute)

        with TempDir() as tmp:
            mlflow.log_artifact("./logs/run.log")

    return report, parameters


def main(annotations, parameters):
    # erase run log
    with open('logs/run.log', 'w'): pass

    # annotations = utils.select(annotations, user=[i for i in range(13) if i not in [2, 4]])

    if parameters["cv"]["active"]:
        A, B = main_with_cv(annotations, parameters)
    if not parameters["cv"]["active"]:
        A, B = main_no_cv(annotations, parameters)

    return A, B
