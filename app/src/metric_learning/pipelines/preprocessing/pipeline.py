from kedro.pipeline import Pipeline, node

from .nodes import rename, add_derivatives, combine

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=rename,
                inputs=["raw_users", "raw_templates"],
                outputs=["renamed_users", "renamed_templates"],
                name="rename_node",
            ),
            node(
                func=add_derivatives,
                inputs=["renamed_users", "renamed_templates", "params:preprocessing.derivatives"],
                outputs=["processed_users", "processed_templates"],
                name="derivative_node",
            ),
            node(
                func=combine,
                inputs=["renamed_users", "renamed_templates", "processed_users", "processed_templates"],
                outputs=["users", "templates"],
                name="combine_node",
            ),
        ],
        tags="intermediate"
    )
