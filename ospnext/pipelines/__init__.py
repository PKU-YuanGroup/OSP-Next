from .t2v_pipeline import pipeline as t2v

pipelines = {}
pipelines.update(t2v)
__all__ = [
    'pipelines'
]