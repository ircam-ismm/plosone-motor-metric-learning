import logging

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import pandas as pd

import sklearn.model_selection as skms

import mlflow
from mlflow.utils.file_utils import TempDir

import pingouin as pg

from metric_learning.extras import utils
from .models import Model


def main_no_cv(users, templates, annotations, parameters):
    """This is meant to optimise full on one dataset and test on the other one.
    The two dataset used here are individual and mean.
    """

    # hyper parameter search
    n_regions = parameters['opt']['n_regions']
    for n_region in n_regions:

        logging.info("start of n_region: {}".format(n_region))

        scale_init = parameters['opt']['init_scale']
        X_init = np.ones(n_region) * scale_init

        model = Model(users, templates, n_region, X_init, parameters)

        experiment_id = parameters['mlflow']['experiment_id']

        X, Y = annotations.iloc[:90], annotations.iloc[90:]
        Xtrain, ytrain = X[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], X['m']
        Xtest, ytest = Y[['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], Y['m']

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(utils.flatten(parameters))
            mlflow.log_param("n_region", n_region)

            # baseline on test
            cc_start = model.score(Xtest, ytest, refit_model=True)
            logging.info("SCORED: {}".format(cc_start))

            # train the model
            model.fit(Xtrain, ytrain)

            # final performance on test
            cc_end = model.score(Xtest, ytest)
            logging.info("FITTED: {}".format(cc_end))


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


def main_with_cv(users, templates, annotations, parameters):
    """Loop over regions and cross validates training with n-repeated k-folds.
    """

    # hyper parameter search
    n_regions = parameters['opt']['n_regions']
    for n_region in n_regions:

        logging.info("start of n_region: {}".format(n_region))

        columns = ['fold_id', 'train_ids', 'test_ids', 'cc_start', 'cc_end']+['w'+str(i) for i in range(n_region)]
        report = pd.DataFrame(columns=columns)

        scale_init = parameters['opt']['init_scale']
        X_init = np.ones(n_region) * scale_init

        model = Model(users, templates, n_region, X_init, parameters)

        experiment_id = parameters['mlflow']['experiment_id']

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(utils.flatten(parameters))
            mlflow.log_param("n_region", n_region)

            # cross validate the experiment
            n_splits = parameters['cv']['n_splits']
            n_repeats = parameters['cv']['n_repeats']
            random_state = parameters['cv']['random_state']

            rkf = skms.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            users_ids = np.array(list(set(annotations['user'])))
            for fold_id, (train_ids, test_ids) in enumerate(rkf.split(users_ids)):

                # reset model weights between folds
                model.weights = model.init_weights

                # save test data for performance measure
                Xtrain, ytrain, Xtest, ytest = train_test_split(annotations, train_ids, test_ids)
                logging.info("LOOP {}: {}".format(fold_id, (train_ids, test_ids)))

                Xtest_mean = utils.select(Xtest, dataset='mean')
                ytest_mean = utils.select(ytest, dataset='mean')

                # baseline on test
                cc_start = model.score(Xtest_mean, ytest_mean['m'], fun='correlation')
                # cc_start = 1-model.score(Xtest_mean, ytest_mean['m'], fun='correlation')
                logging.info("baseline mean: {}".format(cc_start))

                # train the model
                model.fit_with_validation(Xtrain, ytrain['m'], True, True)

                # final performance on test
                cc_end = model.score(Xtest_mean, ytest_mean['m'], fun='correlation')
                # report for current fold
                row = pd.DataFrame(data=[[fold_id, train_ids, test_ids, cc_start, cc_end]+list(model.weights)],
                                columns=columns)

                logging.info("LOOP {}: {:.3f} -> {:.3f}".format(fold_id, cc_start, cc_end))
                report = report.append(row)


            # score model on weights' average and full dataset
            X, y = annotations.iloc[:90][['user', 'day_0', 'rep_0', 'day_1', 'rep_1']], annotations.iloc[:90]['m']
            model.weights = model.init_weights
            cc_init = model.score(X, y, fun='correlation')
            mlflow.log_metric(key="cc_init", value=cc_init)

            weights = report.filter(regex="w.*").mean().values
            model.weights = weights
            cc_all = model.score(X, y, fun='correlation')
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

            # log artifacts
            with TempDir() as tmp:
                file_path = tmp.path("report.csv")
                with open(file_path, 'w') as f: report.to_csv(f)
                mlflow.log_artifact(file_path)

            with TempDir() as tmp:
                file_path = tmp.path("params.txt")
                import json
                with open(file_path, 'w') as f: json.dump(parameters, f)
                mlflow.log_artifact(file_path)

            # with TempDir() as tmp:
            #     file_path = tmp.path("src.txt")
            #     import inspect
            #     import sys
            #     data = inspect.getsource(sys.modules[__name__])
            #     with open(file_path, 'w') as f:
            #         for line in data: f.write(line)
            #     mlflow.log_artifact(file_path)

            import sys
            import metric_learning
            log_source_code("training.py", sys.modules[__name__])
            log_source_code("models.py", metric_learning.pipelines.optimisation.temporal.models)

            with TempDir() as tmp:
                mlflow.log_artifact("./logs/run.log")


    return report, parameters


def log_source_code(logname, modulename):
    with TempDir() as tmp:
        file_path = tmp.path(logname)
        import inspect
        data = inspect.getsource(modulename)
        with open(file_path, 'w') as f:
            for line in data: f.write(line)
        mlflow.log_artifact(file_path)




def main(users, templates, annotations, parameters):
    # erase run log
    with open('logs/run.log', 'w'): pass

    if parameters["cv"]['active']:
        A, B = main_with_cv(users, templates, annotations, parameters)

    if not parameters["cv"]['active']:
        A, B = main_no_cv(users, templates, annotations, parameters)

    return A, B