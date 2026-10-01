"""
Library for processing of DAS data
"""

import numpy as np
import scipy.signal as sps
import xarray as xr
import dask.array as da
from typing import Callable, List

import daspal.core as dp
import daspal.instr.registry as registry

from daspal._logger import logger

#######################################################################################

def blocks(
    blocks: list[xr.DataArray],
    func: Callable,
    *args,
    **kwargs,
) -> list[xr.DataArray]:

    """
    Apply a function independently to each block of a list of DataArrays.

    Parameters
    ----------
        blocks : list[xr.DataArray]
            List of DataArray blocks, for example returned by ``load_data`` when
            temporal gaps are present.

        func : callable
            Function applied to each block. The first argument of ``func`` must
            be a DataArray.

        *args
            Additional positional arguments passed to ``func``.

        **kwargs
            Additional keyword arguments passed to ``func``.

    Returns
    -------
        list[xr.DataArray]
            List of processed DataArrays.
    """

    return [func(block, *args, **kwargs) for block in blocks]


#######################################################################################
############################## Resampling data ########################################
#######################################################################################

def _trim_to_factor(xarr, dim, factor):
    """
    Trim an xarray DataArray along a given dimension so its length
    becomes divisible by `factor`.

    Parameters
    ----------
        xarr : xarray.DataArray
            Input DataArray
        dim : str
            Dimension along which to trim
        factor : int
            Must divide the total length along `dim`

    Returns
    -------
        xarr_trimmed : xarray.DataArray
            Trimmed DataArray (or original if no trimming needed)

    Note
    ----
        Warning message logged describing the trimming, or nothing if no trimming occurred
    """
    size = xarr.sizes[dim]
    extra = size % factor

    if extra:
        xarr = xarr.isel({dim: slice(0, size - extra)})        
        
        logger.warning(f"Dimension '{dim}' trimmed from {size} to {size - extra} "
                       f"to be divisible by factor {factor}")
    
    xarr.attrs = dp.adjust_metadata(xarr)
    
    return xarr
    
#####################################

def _rechunk_to_factor(xarr, dim, factor):
    """
    Rechunk an xarray DataArray so that all chunks along `dim` are divisible by `factor`.
    The last chunk absorbs any remainder data to preserve total length.
    Used in within the `process.resample` function.

    Parameters
    ----------
        xarr : xarray.DataArray
            Input DataArray
        dim : str
            Dimension along which to rechunk
        factor : int
            Must divide the total length along `dim`

    Returns
    -------
        xarray.DataArray
            Rechunked DataArray
    """

    chunks_axis = xarr.chunks[xarr.dims.index(dim)]
    total = sum(chunks_axis)

    # Round down each chunk to nearest multiple of factor
    new_chunks = [(chunk // factor) * factor for chunk in chunks_axis]

    # Compute remainder and add share it with each chunk
    extra = total - sum(new_chunks)
    i = 0
    while extra > 0:
        new_chunks[i] += factor
        extra -= factor
        i = (i + 1) % len(new_chunks) # wrap back to first index over and over   

    return xarr.chunk({dim: tuple(new_chunks)})

#####################################

def _resample_dim(xarr, dim, factor):

    """
    Resample a single xarray.DataArray along a given dimension using scipy resample_poly.
    Handles FIR filter edge effects for Dask arrays.

    Parameters
    ----------
        xarr : xarray.DataArray
            Input DataArray to resample
        dim : str
            Dimension name along which to resample
        factor : int
            Downsampling factor along this dimension

    Returns
    -------
        xarray.DataArray
            Resampled DataArray with updated coordinates
    """

    if factor == 1:
        # No resampling needed
        return xarr

    axis = xarr.dims.index(dim)
    arr = xarr.data
    coord_vals = xarr.coords[dim].values

    if isinstance(arr, da.Array):
        # Dask: apply map_overlap to handle chunk boundaries correctly
        # overlap to handle FIR filter edge effects
        fir_len = 10 * factor        # FIR filter length (resample_poly default)
        overlap_res = int(np.ceil(fir_len / factor))  # trim scaled by factor

        # Wrapper function to resample a single block and trim overlap
        def resample_block(block):
            # Apply resample_poly along the specified axis
            resampled = sps.resample_poly(block, up=1, down=factor, axis=axis)
            # Trim the overlap added for FIR edge effects
            slicer = [slice(None)] * block.ndim
            slicer[axis] = slice(overlap_res, -overlap_res)
            return resampled[tuple(slicer)]

        # update chunks with new chunks along dim
        chunks = list(xarr.chunks)
        new_chunks_axis = tuple(int(np.ceil(chunk * 1 / factor)) for chunk in chunks[axis])
        chunks[axis] = new_chunks_axis

        arr_res = arr.map_overlap(
            resample_block,
            depth={axis: fir_len},  # # extra samples per chunk for FIR edges
            boundary='reflect',     # reflect values at edges
            trim=False,             # trimming handled inside resample_block
            dtype=arr.dtype,
            chunks=chunks  # use downsampled chunks
        )
    else:
        # NumPy array: resample directly
        arr_res = sps.resample_poly(arr, up=1, down=factor, axis=axis)

    # Compute new coordinates along this axis
    new_len = arr_res.shape[axis]

    if dim == 'time':
        new_coords = coord_vals[0] + np.arange(new_len) * np.timedelta64(int(xarr.attrs['dt'] * factor * 1e9), 'ns')
    elif dim == 'distance':
        # Numeric coordinates
        new_coords = coord_vals[0] + np.arange(new_len) * xarr.attrs['dx'] * factor
        # convert distance to float32
        new_coords = new_coords.astype(np.float32, copy=False)

    return xr.DataArray(
        arr_res,
        coords={d: (xarr.coords[d].dims, new_coords if d == dim else xarr.coords[d].values)
                for d in xarr.coords},
        dims=xarr.dims,
        attrs=xarr.attrs
    )
    
#####################################

def resample(
    xarr: xr.DataArray,
    timefactor: int = 1,
    distfactor: int = 1,
) -> xr.DataArray:

    """
    Dask-compatible resampling of a DAS DataArray along ``time`` and
    ``distance`` using ``scipy.signal.resample_poly``.

    Performs downsampling while preserving Dask chunking, handling FIR
    filter edge effects, and updating coordinates and metadata.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.
            Must contain ``dt`` and ``dx`` attributes defining the original
            sampling intervals.

        timefactor : int, default=1
            Downsampling factor along ``time``. A value of 1 disables time
            resampling.

        distfactor : int, default=1
            Downsampling factor along ``distance``. A value of 1 disables
            distance resampling.

    Returns
    -------
        xr.DataArray
            Resampled DataArray with updated ``time`` and ``distance``
            coordinates. Metadata attributes are updated according to the
            downsampling factors.

    Notes
    -----
        The input array is trimmed with a warning if its size along a dimension
        is not divisible by the corresponding downsampling factor, avoiding
        incomplete resampling windows.

    See Also
    --------
        _resample_dim
            Resample a single dimension with Dask support.

        _trim_to_factor
            Trim an array so its size is divisible by a given factor.

        _rechunk_to_factor
            Rechunk a Dask array to match the resampling factor.
    """

    xarr_res = xarr.copy()
    
    # Defines original chunk size for rechunk after resampling
    chunks = {}
    if timefactor > 1:
        chunks["time"] = xarr.chunksizes["time"][0]
    if distfactor > 1:
        chunks["distance"] = xarr.chunksizes["distance"][0]

    
    # --- Trim full array sizes with warning ---
    if xarr_res.sizes['time'] % timefactor != 0:
        xarr_res = _trim_to_factor(xarr_res, 'time', timefactor)
    if xarr_res.sizes['distance'] % distfactor != 0:
        xarr_res = _trim_to_factor(xarr_res, 'distance', distfactor)

    # --- Handle dask chunking ---
    if isinstance(xarr_res.data, da.Array):
        # make chunks divisible by factor
        xarr_res = _rechunk_to_factor(xarr_res, 'time', timefactor)
        xarr_res = _rechunk_to_factor(xarr_res, 'distance', distfactor)
  
    # --- Resample along time dimension ---
    xarr_res = _resample_dim(xarr_res, 'time', timefactor)

    # --- Resample along distance dimension ---
    xarr_res = _resample_dim(xarr_res, 'distance', distfactor)

    # --- Update metadata ---
    xarr_res.attrs['dt'] = xarr.attrs['dt'] * timefactor
    xarr_res.attrs['dx'] = xarr.attrs['dx'] * distfactor
    xarr_res.attrs = dp.adjust_metadata(xarr_res)

    # rechunk after resampling to avoid tiny chunks
    #xarr_res = xarr_res.chunk("auto")
    if chunks:
        xarr_res = xarr_res.chunk(chunks)
   
    return xarr_res

#####################################

def select_channels(
    xarr: xr.DataArray,
    spacing: float = 500,
) -> xr.DataArray:

    """
    Select channels at regular spatial intervals along ``distance``.

    This function is Dask-compatible and selects channels at a given physical spacing
    to reduce the number of channels.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.
            Must contain ``dt`` and ``dx`` attributes defining the original
            sampling intervals.

        spacing : float, default=500
            Desired spacing between selected channels, in the same units as
            ``dx``.

    Returns
    -------
        xr.DataArray
            DataArray downsampled along ``distance`` with updated ``distance``
            coordinates and metadata.

    Notes
    -----
        This function does not apply filtering or anti-aliasing. It is mainly
        intended for large channel spacing reductions.
    """

    xarr_res = xarr.copy()
    
    # compute step in number of channels
    step = int(round(spacing / xarr.attrs['dx']))
    
    # select every `step`-th channel along distance
    xarr_res = xarr_res.isel(distance=slice(0, None, step))
    
    # update metadata
    xarr_res.attrs['dx'] = xarr.attrs['dx'] * step
    xarr_res.attrs = dp.adjust_metadata(xarr_res)
    
    return xarr_res


#######################################################################################
########################### Common mode removal #######################################
#######################################################################################

def cmr(
    xarr: xr.DataArray,
    win: float | None = None,
) -> xr.DataArray:
    """
    Apply Common Mode Removal (CMR) to a DataArray.

    CMR removes the common signal component across channels. A global mean is
    removed when no window length is provided; otherwise, a sliding spatial
    window is used.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        win : float, default=None
            Spatial window length used for the CMR sliding window, in meters.

            If ``None``, the global mean is removed. This applies the same
            correction across all channels. For spatially varying conditions,
            applying CMR on smaller cable sections can be more appropriate.

    Returns
    -------
        xr.DataArray
            CMR-corrected DataArray.

    Notes
    -----
        The choice of window size depends on the cable environment:

        - Flat cable sections may work well with larger windows.
        - Steep or variable bathymetry may require smaller windows.

        However default win=None might be the safer approach: all channels scaled the same way
        but it might be better to apply it directly on smaller cable sections
    """

    attrs = xarr.attrs
    
    if win is None:
        # CMR applied considering all distances
        cmode = xarr.mean(dim='distance', skipna=True)
    else:
        # CMR applied using a sliding window where each channel is correctd
        # by neighbouring channels over a specific section
        dx = xarr.attrs['dx']
        N = int(window / dx)
        cmode = xarr.rolling(distance=N, center=True, min_periods=1).mean(skipna=True)
    
    xarr_cmr = xarr - cmode
    xarr_cmr.attrs = attrs
    return xarr_cmr

 
#######################################################################################
######################### Process data with DAS instr specs ###########################
#######################################################################################

def DASinstr(
    xarr: xr.DataArray,
    instr: str | None = "optodas",
    strain: bool = True,
    unwrap: bool = False,
    integrate: bool = False,
) -> xr.DataArray:

    """
    Pre-process DAS data according to instrument specifications.

    This function applies instrument-specific conversions and corrections.
    It can convert phase measurements to strain, unwrap spatial phase/strain
    variations, and integrate strain-rate data over time.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        instr : str, default="optodas"
            Instrument type used for processing. Currently, only ``"optodas"``
            is supported.

        strain : bool, default=True
            If True, convert phase or phase-rate data to strain or strain-rate.

        unwrap : bool, default=False
            If True, perform spatial unwrapping along the fiber. Only valid when
            data are expressed as strain or strain-rate.

        integrate : bool, default=False
            If True, integrate strain-rate over time to obtain cumulative strain.

    Returns
    -------
        xr.DataArray
            Processed DataArray with ``time`` and ``distance`` coordinates.
    
    Notes
    -----
        It is currently implemented for ``optodas`` instruments only and is
        following the processing approach used in `simpledas 
        <https://github.com/ASN-Norway/simpleDAS>`_.
    """

    # assign instrument
    try:
        instrument = registry.get_instrument(instr)
    except KeyError:
        raise ValueError(
        f"`instr` needs to be defined as: {', '.join(INSTRUMENTS.keys())}\n"
        f"if data already processed with package try 'instr=dasproc'"
        )

    xarr = instrument.process(
        xarr, strain, unwrap, integrate, 
    )
    
    return xarr
