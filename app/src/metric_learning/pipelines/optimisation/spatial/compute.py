import multiprocessing as mp
from multiprocessing import util
from multiprocessing.pool import MaybeEncodingError, ExceptionWithTraceback

def _helper_reraises_exception(ex):
    'Pickle-able helper function for use by _guarded_task_generation.'
    raise ex

# https://stackoverflow.com/questions/740844/python-multiprocessing-pool-of-custom-processes
ctx = mp.get_context()

class MyProcess(ctx.Process):

    # class variables
    distance = 'default'
    radius = 100
    weights_shape = -1
    data_shape = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = self.worker

    # https://github.com/python/cpython/blob/3.9/Lib/multiprocessing/pool.py
    def worker(self, inqueue, outqueue, initializer=None, initargs=(), maxtasks=None,
               wrap_exception=False):
        if (maxtasks is not None) and not (isinstance(maxtasks, int)
                                           and maxtasks >= 1):
            raise AssertionError("Maxtasks {!r} is not valid".format(maxtasks))

        put = outqueue.put
        get = inqueue.get
        if hasattr(inqueue, '_writer'):
            inqueue._writer.close()
            outqueue._reader.close()

        if initializer is not None:
            initializer(*initargs)

        completed = 0
        while maxtasks is None or (maxtasks and completed < maxtasks):
            try:
                task = get()
            except (EOFError, OSError):
                util.debug('worker got EOFError or OSError -- exiting')
                break

            if task is None:
                util.debug('worker got sentinel -- exiting')
                break

            job, i, func, args, kwds = task
            # here we use a custom interface
            try:
                fun = args[0][0]
                if fun == 'compute':
                    result = (True, list(map(self._compute_dtw_row, args[0][1])))
                elif fun == 'reset':
                    result = (True, 0)

                # original interface
                else:
                    result = (True, func(*args, **kwds))

            except Exception as e:
                if wrap_exception and func is not _helper_reraises_exception:
                    e = ExceptionWithTraceback(e, e.__traceback__)
                result = (False, e)


            try:
                put((job, i, result))
            except Exception as e:
                wrapped = MaybeEncodingError(e, result[1])
                util.debug("Possible encoding error while sending result: %s" % (
                    wrapped))
                put((job, i, (False, wrapped)))

            task = job = result = func = args = kwds = None
            completed += 1
        util.debug('worker exiting after %d tasks' % completed)


    def run(self):
        # https://github.com/python/cpython/blob/3.9/Lib/multiprocessing/process.py#L103

        # load the data for each process
        from kedro.framework.session import get_current_session
        session = get_current_session()
        context = session.load_context()
        self.users = context.catalog.load('users')
        self.templates = context.catalog.load('templates')

        # run
        if self._target:
            self._target(*self._args, **self._kwargs)


    def _compute_dtw_row(self, data):
        """Compute the DTW between both observations described in the row and
        the template. Return the difference between the DTW.
        """
        row = data[0][1]
        weights = data[1][0]
        dimensions = data[1][1]

        from metric_learning.extras import utils
        import sklearn.preprocessing as skprep
        import fastdtw2
        import numpy as np

        t = utils.select(self.templates, template=1, version=0)
        t = skprep.StandardScaler().fit_transform(t[list(dimensions)])

        a = utils.select(self.users, gesture=1, user=row['user'], day=row['day_0'], trial=row['rep_0'])
        a = skprep.StandardScaler().fit_transform(a[list(dimensions)])

        b = utils.select(self.users, gesture=1, user=row['user'], day=row['day_1'], trial=row['rep_1'])
        b = skprep.StandardScaler().fit_transform(b[list(dimensions)])


        if self.distance == 'diag':
            # weights are on the diagonals
            da, path_a = fastdtw2.fastdtw(a, t,
                dist="mahalanobis_diag", radius=self.radius, weights_list=weights)
            db, path_b = fastdtw2.fastdtw(b, t,
                dist="mahalanobis_diag", radius=self.radius, weights_list=weights)

        if self.distance == 'full':
            # weights must be placed as full inverse covariance matrix
            # we use an upper triangular matrix of weights, which multiplied by
            # its transpose makes sure we have a semi-positive definite matrix
            # https://stats.stackexchange.com/questions/11368/how-to-ensure-properties-of-covariance-matrix-when-fitting-multivariate-normal-m
            A = np.eye(self.data_shape) * weights[:self.data_shape]
            B = weights[self.data_shape:].reshape(self.data_shape, self.data_shape)
            M = B@A@B.T
            weights = M.reshape(-1)

            da, path_a = fastdtw2.fastdtw(a, t,
                dist="mahalanobis_full", radius=self.radius, weights_list=weights)
            db, path_b = fastdtw2.fastdtw(b, t,
                dist="mahalanobis_full", radius=self.radius, weights_list=weights)


        return da - db
