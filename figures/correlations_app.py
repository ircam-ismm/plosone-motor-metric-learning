from nbimports import *

import sklearn.preprocessing as skprep


# def colors_dimensions(dimensions):
#     colors = sns.color_palette("tab10")
#     colors = np.array([[col]*3 for col in colors[0:10:2]]).reshape(-1, 3)
#     colors_dict = dict(zip(dimensions, colors))
#     return colors_dict

def colors_dimensions(dimensions):
    colors = [sns.color_palette("tab10")[i] for i in [8,9,6,4,5]]
    colors = np.array([[col]*3 for col in colors]).reshape(-1, 3)
    colors_dict = dict(zip(dimensions, colors))
    return colors_dict

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


def compute_mse_diff_temporal(users, templates, dimensions, N_SEGMENTS, signal_fun, agg_fun):

    def fun(row):
        template = select(templates, template=1, version=0)
        a = select(users, gesture=1, user=row["user"], day=row["day_0"], trial=row["rep_0"])
        b = select(users, gesture=1, user=row["user"], day=row["day_1"], trial=row["rep_1"])

        # normalised position
        a = skprep.StandardScaler().fit_transform(a[dimensions])
        b = skprep.StandardScaler().fit_transform(b[dimensions])
        t = skprep.StandardScaler().fit_transform(template[100:][dimensions])

        # vanilla DTW
        da, path_a = fastdtw.fastdtw(a, t, radius=10)
        db, path_b = fastdtw.fastdtw(b, t, radius=10)
        path_a = np.array(path_a)
        path_b = np.array(path_b)

        # ponderate
        weights = np.ones(N_SEGMENTS)
        vec = ponderate(t, np.arange(weights.shape[0]))

        # compute MSE per segment
        t_n = np.linalg.norm( t , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec, t_n]).T, columns=["s", "d"]).astype({"s":int})
        signal_segment = df.groupby("s").agg(signal_fun)["d"].values

        da_ = np.linalg.norm( (a[path_a[:,0]] - t[path_a[:,1]]) , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec[path_a[:, 1]], da_]).T, columns=["s", "d"]).astype({"s":int})
        da_ = df.groupby("s").agg(agg_fun)["d"].values #/ signal_segment

        db_ = np.linalg.norm( (b[path_b[:,0]] - t[path_b[:,1]]) , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec[path_b[:, 1]], db_]).T, columns=["s", "d"]).astype({"s":int})
        db_ = df.groupby("s").agg(agg_fun)["d"].values #/ signal_segment

        row["mse_a"] = da_
        row["mse_b"] = db_
        row["mse_diff"] = da_ - db_

        return row

    return fun


def format_mse_diff_temporal(mse_t, N_SEGMENTS):

    mse_t[[str(i) for i in range(N_SEGMENTS)]] = mse_t["mse_diff"].to_list()
    mse_t[["a_"+str(i) for i in range(N_SEGMENTS)]] = mse_t["mse_a"].to_list()
    mse_t[["b_"+str(i) for i in range(N_SEGMENTS)]] = mse_t["mse_b"].to_list()

    mse_t_a = mse_t.filter(regex="a_.*")
    mse_t_a.columns = [str(i) for i in range(N_SEGMENTS)]
    mse_t_b = mse_t.filter(regex="b_.*")
    mse_t_b.columns = [str(i) for i in range(N_SEGMENTS)]

    mse_t_c = pd.concat([mse_t_a, mse_t_b])

    return mse_t, mse_t_c



def compute_mse_temporal(users, templates, dimensions, N_SEGMENTS, signal_fun, agg_fun):

    def fun(grp):

        template = select(templates, template=1, version=0)

        # normalised position
        a = skprep.StandardScaler().fit_transform(grp[dimensions])
        t = skprep.StandardScaler().fit_transform(template[100:][dimensions])

        # vanilla DTW
        da, path_a = fastdtw.fastdtw(a, t, radius=10)
        path_a = np.array(path_a)

        # ponderate
        weights = np.ones(N_SEGMENTS)
        vec = ponderate(t, np.arange(weights.shape[0]))

        # compute MSE per segment
        t_n = np.linalg.norm( t , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec, t_n]).T, columns=['s', 'd']).astype({"s":int})
        signal_segment = df.groupby("s").agg(signal_fun)['d'].values

        da_ = np.linalg.norm( (a[path_a[:,0]] - t[path_a[:,1]]) , axis=1, ord=1)
        df = pd.DataFrame(data=np.vstack([vec[path_a[:, 1]], da_]).T, columns=['s', 'd']).astype({"s":int})
        da_ = df.groupby("s").agg(agg_fun)['d'].values #/ signal_segment

        return da_

    return fun


def format_mse_temporal(mse_t, N_SEGMENTS):

    mse_t = mse_t.reset_index()
    mse_t[[str(i) for i in range(N_SEGMENTS)]] = mse_t[0].to_list()
    mse_t['x'] = (mse_t['day'] - 1) * 15 + mse_t['trial']

    return mse_t


import fastdtw2

def compute_mse_diff_spatial(users, templates, dimensions, weights_list=None):

    def fun(row):
        template = select(templates, template=1, version=0)
        a = select(users, gesture=1, user=row["user"], day=row["day_0"], trial=row["rep_0"])
        b = select(users, gesture=1, user=row["user"], day=row["day_1"], trial=row["rep_1"])

        # normalised position
        a = skprep.StandardScaler().fit_transform(a[dimensions])
        b = skprep.StandardScaler().fit_transform(b[dimensions])
        t = skprep.StandardScaler().fit_transform(template[100:][dimensions])

        # vanilla DTW or special DTW
        if weights_list is not None:
            da, path_a = fastdtw2.fastdtw(a, t, radius=10, dist="mahalanobis_diag", weights_list=weights_list)
            db, path_b = fastdtw2.fastdtw(b, t, radius=10, dist="mahalanobis_diag", weights_list=weights_list)
        else:
            da, path_a = fastdtw.fastdtw(a, t, radius=10)
            db, path_b = fastdtw.fastdtw(b, t, radius=10)

        path_a = np.array(path_a)
        path_b = np.array(path_b)

        # MSE
        mse_a = ((a[path_a[:,0]] - t[path_a[:,1]])**2).mean(0)
        mse_b = ((b[path_b[:,0]] - t[path_b[:,1]])**2).mean(0)

        row["mse_a"] = mse_a
        row["mse_b"] = mse_b
        row["mse_diff"] = mse_a - mse_b

        return row

    return fun


def format_mse_diff_spatial(mse_s, dimensions):
    dim_len = len(dimensions)

    mse_s[[str(i) for i in range(dim_len)]] = mse_s["mse_diff"].to_list()
    mse_s[["a_"+str(i) for i in range(dim_len)]] = mse_s["mse_a"].to_list()
    mse_s[["b_"+str(i) for i in range(dim_len)]] = mse_s["mse_b"].to_list()

    mse_a = mse_s.filter(regex="a_.*")
    mse_a.columns = dimensions
    mse_b = mse_s.filter(regex="b_.*")
    mse_b.columns = dimensions
    mse_c = pd.concat([mse_a, mse_b])
    mse_c_long = mse_c.melt(value_vars=dimensions)

    return mse_s, mse_c_long


def compute_mse_spatial(users, templates, dimensions, weights_list=None):

    def fun(grp):

        template = select(templates, template=1, version=0)

        a = skprep.StandardScaler().fit_transform(grp[dimensions])
        t = skprep.StandardScaler().fit_transform(template[100:][dimensions])

        if weights_list is not None:
            da, path_a = fastdtw2.fastdtw(a, t, radius=10, dist="mahalanobis_diag", weights_list=weights)
        else:
            da, path_a = fastdtw2.fastdtw(a, t, radius=10)

        path_a = np.array(path_a)
        mse_a = ((a[path_a[:,0]] - t[path_a[:,1]])**2).mean(0)
        return mse_a

    return fun

def format_mse_spatial(mse_s, N_DIMS):
    mse_s = mse_s.reset_index()
    mse_s[[str(i) for i in range(N_DIMS)]] = mse_s[0].to_list()
    mse_s['x'] = (mse_s['day'] - 1) * 15 + mse_s['trial']
    return mse_s
