import logging
log = logging.getLogger(__name__)

import pandas as pd
import sklearn.preprocessing as skprep
import scipy.stats as sstats

import fastdtw

from metric_learning.extras import utils

import mlflow
from mlflow.utils.file_utils import TempDir

def compute(users, templates, annotations, parameters):

    with mlflow.start_run(experiment_id=parameters['mlflow']['experiment_id']):

        # erase run log
        with open('logs/run.log', 'w'): pass



        dims = parameters['data']['dims']
        normalise = parameters['data']['normalise']
        radius = parameters['dtw']['radius']

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

            da, path_a = fastdtw.fastdtw(a, template, radius=radius)
            db, path_b = fastdtw.fastdtw(b, template, radius=radius)
            row['DTW'] = da - db

            return row

        res = annotations.apply(compute_, axis=1)
        cor = sstats.linregress(res['DTW'].values, annotations['m'].values)
        log.info("Result: {} r:{:.2f} p:{:.1e}".format(dims, cor.rvalue, cor.pvalue))

        # prepare report
        report = pd.DataFrame(data=[[",".join(dims), radius, normalise, *list(cor)]],
                            columns=['dims', 'radius', 'normalise', *cor._fields])


        with TempDir() as tmp:
            # file_path = tmp.path("log.txt")
            # with open(file_path, 'w') as f:
            #     pd.DataFrame([cor], columns = cor._fields).to_csv(f)
            mlflow.log_artifact("./logs/run.log")



    return report


from kedro.pipeline import Pipeline, node
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=compute,
                inputs=["users", "templates", "annotations_90_repeat", "parameters"],
                outputs="exp1_report",
                name="exp1_correlation_node",
            ),
        ],
        tags="compute"
    )
