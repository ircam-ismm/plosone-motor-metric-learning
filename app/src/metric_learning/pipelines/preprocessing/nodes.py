import pandas as pd
import numpy as np
import scipy.signal as ss

from typing import Tuple

def rename(users: pd.DataFrame,
           templates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rename columns from the pandas dataframes.
    Add derivatives (and norm) for position data.
    """
    rename_dict = {"id":"user",
                   "position_x": "px", "position_y": "py", "position_z":"pz",
                   "accel_x": "ax", "accel_y": "ay", "accel_z": "az",
                   "gyro_x" : "gx", "gyro_y" : "gy", "gyro_z": "gz",
                   "quart_1": "q1", "quart_2": "q2", "quart_3": "q3", "quart_4": "q4",
                   "orient_x": "ox", "orient_y": "oy", "orient_z": "oz",}

    users = users.rename(columns=rename_dict)
    templates = templates.rename(columns=rename_dict)

    return users, templates

def add_derivatives(users: pd.DataFrame,
                    templates: pd.DataFrame,
                    derivatives: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add derivatives of the position data with Savitzky-Golay interpolation.
    """
    data_dims = ['px', 'py', 'pz', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    usrs = users[['user', 'day', 'gesture', 'trial', 'sample']+data_dims].copy()
    tmps = templates[['template', 'version']+data_dims].copy()

    def _process(grp):
        # filter position, accelerometer, gyroscope
        for sensor in 'pag':
            for axis in 'xyz':
                col = sensor+axis
                grp[col+'0'] = ss.savgol_filter(grp[col], window_length=15, polyorder=3, deriv=0)

        # differenciate from filtered position
        for axis in 'xyz':
            for der in range(derivatives):
                basis = 'p'+axis+str(der)
                deriv = 'p'+axis+str(der+1)
                grp[deriv] = ss.savgol_filter(grp[basis], window_length=15, polyorder=3, deriv=1)

        return grp

    usrs = usrs.groupby(['user', 'day', 'trial']).apply(_process)
    tmps = tmps.groupby(['template', 'version']).apply(_process)

    return usrs, tmps

def combine(renamed_users: pd.DataFrame,
            renamed_templates: pd.DataFrame,
            processed_users: pd.DataFrame,
            processed_templates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-combine the renamed dataframes and the newly processed dataframes.
    """
    acc_gyr = ['ax0', 'ay0', 'az0', 'gx0', 'gy0', 'gz0']
    pos = ['p'+axis+deriv for axis in 'xyz' for deriv in '012']

    users = pd.concat([renamed_users, processed_users[pos+acc_gyr]], axis=1)
    templates = pd.concat([renamed_templates, processed_templates[pos+acc_gyr]], axis=1)

    return users, templates

#     def add_norm(grp):
#         cols = [(der, ['p'+axis+str(der) for axis in 'xyz']) for der in range(n_derivatives)]
#         for i, col in cols:
#             grp['p'+str(i)+'_n'] = np.linalg.norm(grp[col], axis=1)
#         return grp
#     obs = obs.groupby(['user', 'day', 'trial']).apply(add_norm)
#     tmps = tmps.groupby(['template', 'version']).apply(add_norm)
#     tmp = add_norm(tmp)

#     def add_tangential(grp):
#         for col in ['p'+str(i)+'_n' for i in range(n_derivatives)]:
#             grp[col[:-2]+'_t'] = ssig.savgol_filter(grp[col], window_length=15, polyorder=3, deriv=1)
#         return grp
#     obs = obs.groupby(['user', 'day', 'trial']).apply(add_tangential)
#     tmps = tmps.groupby(['template', 'version']).apply(add_tangential)
#     tmp = add_tangential(tmp)

# whiten
# stdscaler = skprep.StandardScaler()
# data = template[['px', 'py', 'pz']]
# _=stdscaler.fit(data)
# data_ = pd.DataFrame(data = stdscaler.transform(data), columns=data.columns)


# import itertools
# def dimensions_combination(dims):
#     dimensions_list = []
#     for i in np.arange(len(dims)):
#         combs = itertools.combinations(dims, i+1)
#         dimensions_list.append(list(combs))
#     dimensions_list = list(itertools.chain(*dimensions_list))
#     return dimensions_list