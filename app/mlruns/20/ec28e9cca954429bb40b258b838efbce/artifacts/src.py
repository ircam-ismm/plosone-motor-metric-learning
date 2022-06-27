import logging
log = logging.getLogger(__name__)
import multiprocessing as mp
import ctypes

import numpy as np
import pandas as pd
from pandarallel import pandarallel
import sklearn.preprocessing as skprep
import scipy.stats as sstats
import fastdtw

from metric_learning.extras import utils

import mlflow
from mlflow.utils.file_utils import TempDir

def compute(users, templates, annotations, parameters):

    parallel = parameters['compute']['parallel']
    nb_workers = parameters['compute']['nb_workers']
    use_memory_fs = parameters['compute']['use_memory_fs']

    pandarallel.initialize(nb_workers=nb_workers)
    if False:
        mparr = mp.Array(ctypes.c_double, users.values.reshape(-1))
        users_dict = dict(list(zip(users.columns, users.dtypes)))
        users = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(users.shape),
                                columns=users.columns).astype(users_dict)

    normalise_list = parameters['data']['normalise']
    dims_list = parameters['data']['dims']
    radius_list = parameters['dtw']['radius']

    # # we do all combinations of dimensions here.
    # import itertools

    # dims_map = {'A': ['p'+axis+'0' for axis in 'xyz'],
    #             'B': ['a'+axis+'0' for axis in 'xyz'],
    #             'C': ['g'+axis+'0' for axis in 'xyz'],
    #             'D': ['p'+axis+'1' for axis in 'xyz'],
    #             'E': ['p'+axis+'2' for axis in 'xyz'],
    #         }
    # dims_combinations = [i for j in range(1, 6) for i in itertools.combinations("ABCDE", j)]

    # dims_list = []
    # for c in dims_combinations:
    #     res = [dims_map[a] for a in c]
    #     dims_list.append((list(itertools.chain.from_iterable(res))))


    for dims in dims_list:

        for normalise in normalise_list:
            for radius in radius_list:

                with mlflow.start_run(experiment_id=parameters['mlflow']['experiment_id']):

                    param_dict = utils.flatten(parameters)
                    param_dict.pop("data.dims", None)
                    param_dict.pop("data.normalise", None)
                    param_dict.pop("dtw.radius", None)

                    mlflow.log_param("data.dims", dims)
                    mlflow.log_param("data.normalise", normalise)
                    mlflow.log_param("dtw.radius", radius)

                    mlflow.log_params(param_dict)


                    dist = parameters['dtw']['dist']
                    averaged = parameters['dtw']['averaged']

                    template = utils.select(templates, template=1, version=0)
                    template = template[dims].iloc[100:]
                    if normalise:
                        template = skprep.StandardScaler().fit_transform(template)

                    def compute_(row):
                        # select correct gesture from dataset
                        a = utils.select(users, gesture=1, user=row['user'], day=row['day_0'], trial=row['rep_0'])
                        b = utils.select(users, gesture=1, user=row['user'], day=row['day_1'], trial=row['rep_1'])
                        # select dimensions
                        a = a[dims]
                        b = b[dims]

                        if normalise:
                            a = skprep.StandardScaler().fit_transform(a)
                            b = skprep.StandardScaler().fit_transform(b)

                        da, path_a = fastdtw.fastdtw(a, template, radius=radius, dist=dist)
                        db, path_b = fastdtw.fastdtw(b, template, radius=radius, dist=dist)

                        if averaged:
                            da = da/a.shape[0]
                            db = db/b.shape[0]

                        row['DTW'] = da - db

                        return row

                    if parallel:
                        res = annotations.parallel_apply(compute_, axis=1)
                    else:
                        res = annotations.apply(compute_, axis=1)

                    X, y = res['DTW'], annotations['m']


                    # correlation for mean
                    Xmean = X.iloc[:90]
                    ymean = y.iloc[:90]
                    cor = sstats.linregress(Xmean.values, ymean.values)
                    mlflow.log_metric(key="r-mean", value=cor.rvalue)
                    mlflow.log_metric(key="p-mean", value=cor.pvalue)

                    # correlation for individual
                    Xind = X.iloc[90:]
                    yind = y.iloc[90:]
                    cor = sstats.linregress(Xind.values, yind.values)
                    mlflow.log_metric(key="r-ind", value=cor.rvalue)
                    mlflow.log_metric(key="p-ind", value=cor.pvalue)

                    # correlation for all
                    cor = sstats.linregress(X.values, y.values)
                    mlflow.log_metric(key="r-all", value=cor.rvalue)
                    mlflow.log_metric(key="p-all", value=cor.pvalue)


                    with TempDir() as tmp:
                        file_path = tmp.path("linregress.csv")
                        with open(file_path, 'w') as f:
                            pd.DataFrame([cor], columns = cor._fields).to_csv(f)
                        mlflow.log_artifact(file_path)

                    with TempDir() as tmp:
                        file_path = tmp.path("compute.csv")
                        with open(file_path, 'w') as f: res.to_csv(f)
                        mlflow.log_artifact(file_path)

                    with TempDir() as tmp:
                        file_path = tmp.path("linregress.png")
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots()
                        sns.regplot(x=res['DTW'].values, y=annotations['m'])
                        fig.savefig(file_path)
                        mlflow.log_artifact(file_path)

                    with TempDir() as tmp:
                        file_path = tmp.path("src.py")
                        import inspect
                        import sys
                        data = inspect.getsource(sys.modules[__name__])
                        with open(file_path, 'w') as f:
                            for line in data: f.write(line)
                        mlflow.log_artifact(file_path)

                    log.info("Result: {}_{}_{} r:{:.2f} p:{:.1e}".format(normalise, dims, radius, cor.rvalue, cor.pvalue))
    return True

from metric_learning.pipelines import select_annotations


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
                func=compute,
                inputs=["users", "templates", "annotations", "parameters"],
                outputs="True",
                name="exp2_correlation_node",
            ),
        ],
        tags="compute"
    )
