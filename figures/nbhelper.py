from nbimports import *

def load_data():
    users = pd.read_csv("../app/data/users.df")
    templates = pd.read_csv("../app/data/templates.df")

    A = pd.read_csv("../app/data/annotations_90_repeat.csv", index_col=0)
    B = pd.read_csv("../app/data/annotations_180_individual.csv", index_col=0)
    columns = ['user', 'day_0', 'rep_0', 'day_1', 'rep_1', 'm']
    annotations = pd.concat([A[columns], B[columns]], ignore_index=True)

    return users, templates, annotations, A[columns], B[columns]


import multiprocessing as mp
import ctypes
def make_parallel_df(df):
    mparr = mp.Array(ctypes.c_double, df.values.reshape(-1))
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))
    df_mp = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape),
                    columns=df.columns).astype(df_dtypes_dict)
    return df_mp


import mlflow
def get_report_params_from_runid(run_id, report_name="report.csv"):
    mc = mlflow.tracking.MlflowClient(tracking_uri="../app/mlruns")
    run = mc.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    report = pd.read_csv(artifact_uri+"/"+report_name, index_col=0)
    params = run.data.params
    return report, params