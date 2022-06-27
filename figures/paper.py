import pandas as pd


def load_data():
    annot_mean = pd.read_csv("../app/data/annotations_90_repeat.csv", index_col=0)
    annot_mean_melt = annot_mean.melt(value_vars=['0', '1', '2', '3'], value_name='judge rating', 
                                  id_vars='m', var_name='judge')
    annot_mean_melt = annot_mean_melt.rename(columns={'m': 'averaged rating'})

    datadir = "../app/data/"
    filenames = ['g0_4.txt', 'g0_4_1.txt', 'g0_4_2.txt']
    dfs = []
    for filename in filenames:
        with open(datadir+filename) as f:
            df = pd.read_json(f, orient="index").sort_index()
            df['choice_'] = df['choice'] / 10 * 2  - 1
            dfs.append(df)
    annot_repeat = pd.concat([df['choice_'] for df in dfs], axis=1)
    annot_repeat.columns = ['0', '1', '2']
    annot_repeat['m'] = annot_repeat.mean(1)
    annot_repeat['std'] = annot_repeat.std(1)
    
    annot_repeat_melt = annot_repeat.melt(value_vars=['0', '1', '2'], value_name='rating', id_vars='m', var_name='repetition')
    annot_repeat_melt = annot_repeat_melt.rename(columns={'m': 'averaged rating'})
    
    return annot_mean, annot_mean_melt, annot_repeat, annot_repeat_melt
    
import matplotlib.pyplot as plt
import seaborn as sns
    
def figure3(annot_mean_melt, annot_repeat_melt):
    fig, axs = plt.subplots(2, 1, figsize=(18,6))
    sns.set_context("paper", font_scale=2)

    ax = axs[0]
    sns.regplot(data=annot_mean_melt, y="judge rating", x="averaged rating", color='k', scatter=False, ax=ax)
    sns.scatterplot(data=annot_mean_melt.sample(frac=1), y="judge rating", x="averaged rating", hue='judge', s=100, ax=ax)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.3, 1.3])
    ax.grid()

    ax.set_xlabel('Averaged rating')
    ax.set_ylabel('Rating')
    ax.legend(loc='upper right', title='Judge')

    ax = axs[1]
    sns.regplot(data=annot_repeat_melt, y="rating", x="averaged rating", color='k', scatter=False, ax=ax)
    sns.scatterplot(data=annot_repeat_melt, x='averaged rating', y='rating', hue='repetition', s=100, ax=ax)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.3, 1.3])
    ax.grid()

    ax.set_xlabel('Averaged rating')
    ax.set_ylabel('Rating')
    ax.legend(loc='upper right', title='Repetition')

    fig.tight_layout()
    return fig


import pingouin as pg
import numpy as np
import scipy
import scipy.stats as scsta

def stats(annot_repeat, annot_mean):
    data = annot_repeat[[str(i) for i in range(3)]].unstack().reset_index()
    data.columns = ['judge', 'motions', 'score']
    
    A = pg.intraclass_corr(data=data, targets='motions', raters='judge', ratings='score')
    res = pg.intraclass_corr(data=data, targets='motions', raters='judge', ratings='score')
    m, ci, df1 = res.iloc[5][["ICC", "CI95%", "df1"]]
    std = np.sqrt(df1) * np.diff(ci) / 3.92
    stats0 = (m, std, df1)
    
    data = annot_mean[[str(i) for i in range(4)]].unstack().reset_index()
    data.columns = ['judge', 'motions', 'score']
    
    B = pg.intraclass_corr(data=data, targets='motions', raters='judge', ratings='score')
    res = pg.intraclass_corr(data=data, targets='motions', raters='judge', ratings='score')
    m, ci, df1 = res.iloc[5][["ICC", "CI95%", "df1"]]
    
    std = np.sqrt(df1) * np.diff(ci) / 3.92
    stats1 = (m, std, df1)
    
    C = scsta.ttest_ind_from_stats(*stats0, *stats1, equal_var=False)
    
    return A, B, C