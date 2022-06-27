
from functools import reduce
from operator import and_, or_

def select(df, **kwargs):
    '''Builds a boolean array where columns indicated by keys in kwargs are
    tested for equality to their values. In the case where a value is a list, a
    logical or is performed between the list of resulting boolean arrays.
    Finally, a logical and is performed between all boolean arrays.
    '''

    res = []
    for k, v in kwargs.items():

        # TODO: if iterable, expand with list(iter)

        # multiple column selection with logical or
        if isinstance(v, list):
            res_or = []
            for w in v:
                res_or.append(df[k] == w)
            res_or = reduce(lambda x, y: or_(x,y), res_or)
            res.append(res_or)

        # single column selection
        else:
            res.append(df[k] == v)

    # logical and
    if res:
        res = reduce(lambda x, y: and_(x,y), res)
        res = df[res]
    else:
        res = df

    return res


from collections import abc
# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='.'):
    """Flattens a nested dictionnary (or else?) into a 1-depth dictionnary by concatenating
    the keys with sep.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


import functools
import weakref
# https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-class-methods-release-object
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator
