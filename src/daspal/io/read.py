import os

from dask import delayed
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import h5py

import daspal.instr.base as base
import daspal.io.h5fc as h5fc

#######################################################################################

def get_dist_slice(distance: np.ndarray, dist_sel: tuple | list | None ):
    """
    Return a slice corresponding to a distance interval [dmin, dmax).
    
    Parameters
    ----------
    distance : np.ndarray
        1D sorted distance array
    dist_sel : list or tuple of length 2, optional
        [dmin, dmax). Use None for no limit.
        
    Returns
    -------
    slice
    """
    # normalize input
    if dist_sel is None:
        dmin, dmax = None, None
    else:
        dmin, dmax = dist_sel

    # compute indices
    idmin = 0 if dmin is None else np.searchsorted(distance, dmin, side='left')
    idmax = len(distance) if dmax is None else np.searchsorted(distance, dmax, side='left')

    return slice(idmin, idmax)


def file_info(fname: str, dist_sel: tuple | list | None, instrument: base.Instr):
    info = instrument.read_data_info(fname)
    dx = info["dx"]
    nx = info["dshape"][1]
    
    distance_axis = np.arange(nx) * dx # this part is probably redundant from read_batch
    dist_slice = get_dist_slice(distance_axis, dist_sel)

    ratio = (dist_slice.stop - dist_slice.start)/nx
    file_size = os.path.getsize(fname)/ (1024 ** 2) # to get mb format (1024 bytes in 1 kilobyte)
    size_read_per_file = ratio*file_size # in Mb

    return info, file_size, size_read_per_file


def _read_batch(fnames: list[str], dist_slice: slice | None,
                instrument: base.Instr, nt_expected: int | None = None) -> np.ndarray:
    """
    read multiple files within one task.
    """
    data = [instrument.read_data(fname, dist_slice) for fname in fnames]
    data = np.concatenate(data, axis=0) # axis 0 being time

    # safety error raising when running with lazy_read_boost=True
    if nt_expected is not None and data.shape[0] != nt_expected:
        raise ValueError(
            f"{nt_expected} time samples expected but {data.shape[0]} read. "
            "The files may not all have the same time length. "
            "Run with lazy_read_boost=False."
        )
    return data


def files_batch(fnames: list[str], read_info: dict) -> xr.DataArray:
    """
    read files batch
    """
    #### parameters
    dist_sel = read_info["dist_sel"]
    instrument = read_info["instrument"]
    lazy_read_boost = read_info["lazy_read_boost"]

    #### gen distance axis and slice ####
    nx_total = read_info["dshape"][1]
    distance_axis = read_info["d0"] + np.arange(nx_total) * read_info["dx"] #float64
    if dist_sel is None:
        nx = nx_total
        dist_slice = slice(0, nx)
    else:
        dist_slice = get_dist_slice(distance_axis, dist_sel)
        distance_axis = distance_axis[dist_slice]
        nx = len(distance_axis)
    # convert distance to float32
    distance_axis = distance_axis.astype(np.float32, copy=False)

    #### handling time axis ####
    # This includes a reading optimization option
    # A- # Assume all files have the same time length as the first file
    if lazy_read_boost:
        # Trust first file time shape already stored in meta_info for every file
        # time axis is then defined outside of this function
        nt = read_info["dshape"][0]*len(fnames)
        coords={'distance': distance_axis}
    # B- Inspect files for correct time length
    else:
        # Inspect files
        nt = 0
        for i, fname in enumerate(fnames):
            f_info = instrument.read_data_info(fname)
            if i == 0:
                t0 = f_info["t0"]
            nt += f_info["dshape"][0]

        time_axis = t0 + pd.timedelta_range(
            start="0s", periods=nt,
            freq="%dns" % int(read_info["dt"] / np.timedelta64(1, "ns") + 0.5)
            )
        coords={'time': time_axis, 'distance': distance_axis}
    
    #### reading data ####
    # "single delayed per batch" -> smaller task graph
    nt_expected = nt if lazy_read_boost else None
    delayed_array = delayed(_read_batch)(fnames, dist_slice, instrument, nt_expected)
    data = da.from_delayed(delayed_array, shape=(nt, nx), dtype=read_info["dtype"])

    #### gen DataArray output ####
    array = xr.DataArray(data=data, 
                         coords=coords,
                         dims=["time", "distance"])
    array.name = 'data'
   
    return array

#############################################################################
##### metadata

def metadata(fname: str, instrument: base.Instr):
    """
    read metadata into a dictionnary
    """
    with h5py.File(fname, 'r') as file:
        m = h5fc.load_h5fields(file, skip=instrument.meta_exclude) # skip data

    return m

