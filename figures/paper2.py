from nbimports import *
mc = mlflow.tracking.MlflowClient(tracking_uri="../app/mlruns")

def create_dims_string(dims):
    ds = []
    
    ap = 'p0,' if 'px0' in dims else '  ,'
    ds.append(ap)
    ap = 'a0,' if 'ax0' in dims else '  ,'
    ds.append(ap)
    ap = 'g0,' if 'gx0' in dims else '  ,'
    ds.append(ap)
    ap = 'p1,' if 'px1' in dims else '  ,'
    ds.append(ap)
    ap = 'p2,' if 'px2' in dims else '  '
    ds.append(ap)
    
    ds = "".join(ds)
    return ds


import sklearn.model_selection as skms

def cross_validate(report):
    n_splits = 2
    n_repeats = 8
    random_state = 42

    rkf = skms.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    users_ids = np.array(list(set(report['user'])))
    data = report.iloc[:90]
    
    ccs = []
    for fold_id, (train_ids, test_ids) in enumerate(rkf.split(users_ids)):
        subset = select(data, user=list(users_ids[train_ids]))
        ccs.append(np.corrcoef(subset['DTW'], subset['m'])[0,1])
    
    return ccs


def getrun():
    res = []

    for run_info in mc.list_run_infos("20"):
        run = mc.get_run(run_info.run_id)

        artifact_uri = run.info.artifact_uri
        report = pd.read_csv(artifact_uri+"/compute.csv", index_col=0)

        dims = create_dims_string(run.data.params['data.dims'])
        rmean = run.data.metrics['r-mean']
        pmean = run.data.metrics['p-mean']

        ccs = np.array(cross_validate(report))
        res.append((dims, rmean, pmean, ccs.mean(), ccs.std(), ccs))

    df = pd.DataFrame(data=res, columns=['dims', 'r', 'p', 'r_m', 'r_s', 'ccs'])

    pd.options.display.float_format = '{:.3f}'.format # None to reset

    return df

def anova(df):
    group_1 = [create_dims_string(i) for i in ["px0", "ax0", "gx0", "px1", "px2"]]
    group_2 = [create_dims_string(i) for i in [["px0", "ax0", "gx0"], ["px0", "px1", "px2"]]]
    group_3 = [create_dims_string(i) for i in [["px0", "ax0", "gx0", "px1", "px2"]]]
    grp = select(df, dims=group_1+group_2+group_3)
    
    ccs = np.array(grp['ccs'].to_list())
    ccs = pd.DataFrame(ccs)
    ccs['cond'] = ccs.index
    ccs_melt = pd.melt(ccs, id_vars='cond')
    
    return pg.anova(data=ccs_melt, dv="value", between="cond")


def twoplots():
    mc = mlflow.tracking.MlflowClient(tracking_uri="../app/mlruns")
    run_id_0 = "ab994aff17da47b9a74f7472084862a2"
    run_id_1 = "a6c2c495c8474651a83495d29d4f46a0"
    
    reports = []
    for run_id in [run_id_0, run_id_1]:
        run = mc.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        reports.append(pd.read_csv(artifact_uri+"/compute.csv", index_col=0))
        # dims = create_dims_string(run.data.params['data.dims'])
        
    from matplotlib.ticker import FormatStrFormatter
    
    data = reports[0].iloc[:90]
    
    sns.set_context("paper", font_scale=2)
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    ax = axs[0]
    sns.scatterplot(data=reports[0].iloc[:90], x='DTW', y='m', hue='user', palette='tab20', s=100, legend=False, ax=ax)
    # sns.scatterplot(data=reports[0].iloc[90:], x='DTW', y='m', color='black', alpha=0.3, ax=ax)
    sns.regplot(data=reports[0].iloc[:90], x='DTW', y='m', color='k', scatter=False, ax=ax)
    ax.set_title(r"R = 0.495")
    ax.set_xlabel("$m_{(p0)}$")
    ax.set_ylabel("$m_{annotation}$")
    # ax.set_xlim([-1050, 1050])

    ax = axs[1]
    sns.scatterplot(data=reports[1].iloc[:90], x='DTW', y='m', hue='user', palette='tab20', s=100, ax=ax)
    # sns.scatterplot(data=reports[1].iloc[90:], x='DTW', y='m', color='black', alpha=0.3, ax=ax)
    g = sns.regplot(data=reports[1].iloc[:90], x='DTW', y='m', color='k', scatter=False, ax=ax)
    ax.set_title(r"R = 0.759")
    ax.set_xlabel("$m_{(acc, gyr, p0, p1, p2)}$")
    ax.set_ylabel("")

    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    for ax in axs:
        ax.grid()
        ax.set_yticks(np.arange(-1, 1.1, .5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylim([-1.1, 1.1])    
    # from matplotlib.ticker import FormatStrFormatter
    # for axis in ax: 
    #     axis.grid()
    #     axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    # ax.plot(np.arange(-4000, 4000, 100), y_pred, 'k')
    fig.tight_layout()