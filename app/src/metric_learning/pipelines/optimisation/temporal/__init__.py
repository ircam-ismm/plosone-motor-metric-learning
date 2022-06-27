from kedro.pipeline import Pipeline, node

from metric_learning.pipelines import select_annotations
from .training import main

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
                func=main,
                inputs=["users", "templates", "annotations", "parameters"],
                outputs=["report", "config"],
                name="temporal_correlation_node"
            ),
        ],
        tags="compute"
    )
