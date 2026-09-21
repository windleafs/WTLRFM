from . import geometry as G
from .das import (build_condition, condition_channels, das_complex, das_groups,
                  resample_target)


def __getattr__(name):
    if name in ("OpenBreastSoSDataset", "load_cached_sample", "load_split"):
        from . import obpw_dataset
        return getattr(obpw_dataset, name)
    raise AttributeError(name)

__all__ = ["G", "build_condition", "condition_channels", "das_complex",
           "das_groups", "resample_target", "OpenBreastSoSDataset",
           "load_cached_sample", "load_split"]
