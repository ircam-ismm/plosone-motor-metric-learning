import pandas as pd

def select_annotations(parameters):
    """This node select the annotations dataset from the parameter 'annotations'.
    The possible values are currently: 'all', 'mean' or 'individual'.
    """
    # fetch the dataset through the session/context/catalog
    # https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/03_session.html
    from kedro.framework.session import get_current_session
    session = get_current_session()
    context = session.load_context()

    A = context.catalog.load('annotations_90_repeat')
    A['dataset'] = 'mean'

    B = context.catalog.load('annotations_180_individual')
    B['dataset'] = 'ind'

    columns = ['user', 'day_0', 'rep_0', 'day_1', 'rep_1', 'm', 'dataset']

    if parameters['annotations'] == 'all':
        annotations = pd.concat([A[columns], B[columns]], ignore_index=True)
    if parameters['annotations'] == 'mean':
        annotations = A
    if parameters['annotations'] == 'individual':
        annotations = B

    return annotations
