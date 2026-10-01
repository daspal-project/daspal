import h5py
import numpy as np

#######################################################################################

def load_h5fields(group, skip=None):
    """
    Recursively load HDF5 group/ file into a dict.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group or file to read from.
    skip : list of dataset/group names to skip
    """
    m = dict(group.attrs)

    for key, val in m.items():
        if isinstance(val, bytes):
            m[key] = val.decode("utf-8")

    for key, item in group.items():
        if skip and key in skip:
            continue
        if isinstance(item, h5py.Group):
            m[key] = load_h5fields(item, skip=skip)
        else:
            val = item[()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            elif isinstance(val, np.ndarray) and (val.dtype.type is np.bytes_ or val.dtype == object):
                val = [x.decode("utf-8") if isinstance(x, bytes) else x for x in val]
            m[key] = val
    return m


def save_h5fields(group, m):
    """
    Save dict into HDF5 group/ file recursively, 
    converting everything into datasets (including scalars/strings).
    
    Parameters
    ----------
    group : h5py.Group
        The HDF5 group or file to write into.
    m : dict
        Dictionary containing data, metadata, nested dicts, arrays, or scalars.
    """
    for key, val in m.items():
        if isinstance(val, dict):
            subgrp = group.create_group(key)
            save_h5fields(subgrp, val)
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val)
        elif isinstance(val, list) and all(isinstance(x, str) for x in val):
            group.create_dataset(key, data=val, dtype=h5py.string_dtype('utf-8'))
        elif isinstance(val, str):
            # write string as dataset
            group.create_dataset(key, data=np.array(val, dtype=h5py.string_dtype('utf-8')))
        else:
            # write scalar as dataset
            group.create_dataset(key, data=val)


def h5file_info(file_path):
    """
    Print HDF5 file structure with:
      - Groups
      - Datasets (shape + dtype + preview)
      - Attributes (with values)

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    max_preview : int
        Max number of elements to preview for datasets.
    """

    def _print(name, obj, indent=0):
        prefix = "  " * indent

        # ---- GROUP ----
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{name}")

            # Print group attributes
            for attr_key, attr_val in obj.attrs.items():
                print(f"{prefix}{attr_key} [ATTR] = {attr_val}")

            # Recurse into children
            for key, item in obj.items():
                child_name = f"{name}/{key}" if name != "/" else f"/{key}"
                _print(child_name, item, indent + 1)

        # ---- DATASET ----
        elif isinstance(obj, h5py.Dataset):
            # Preview small datasets
            try:
                if obj.shape == ():
                    preview = obj[()]
                else:
                    preview = f"shape={obj.shape}, dtype={obj.dtype}"
            except Exception:
                preview = "<cannot preview>"

            print(f"{prefix}{name} [DATASET] = {preview}")

            # Print dataset attributes
            for attr_key, attr_val in obj.attrs.items():
                print(f"{prefix}{attr_key} [ATTR] = {attr_val}")

    with h5py.File(file_path, "r") as f:
        _print("/", f)