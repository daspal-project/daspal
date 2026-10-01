import numpy as np

#######################################################################################

def meta_comp(meta):
    """
    Convert xarray attributes into Zarr-compatible JSON metadata.

    Zarr stores attributes as JSON metadata, so NumPy arrays and
    NumPy scalar types must be converted.

    Parameters
    ----------
    meta : xarray DataArray attributes

    Returns
    -------
    meta : JSON-compatible attributes
    """

    if isinstance(meta, np.ndarray):
        # convert numpy array to Python list
        return meta.tolist()
    elif isinstance(meta, np.generic):
        # convert numpy scalar to Python value
        return meta.item()
    elif isinstance(meta, dict):
        # apply function recursively inside dictionaries
        return {k: meta_comp(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        # apply recursively inside lists
        return [meta_comp(v) for v in meta]
    else:
        # already JSON-compatible
        return meta