"""
Library for signal processing of DAS data
"""

import numpy as np
import pandas as pd
import scipy.signal as sps
import xarray as xr
from functools import partial
import copy
from collections import OrderedDict

import dask.array as da

import daspal.dask_utils.tasks as tasks

from daspal._logger import logger

###############################################################################

def taper(
    xarr: xr.DataArray,
    w_type: str = "tukey",
    alpha: float = 0.1,
    dim: str = "time",
) -> xr.DataArray:

    """
    Apply a tapering window to a DAS DataArray along a given dimension.

    This function is Dask-compatible and applies a windowing function to
    reduce edge effects in the selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        w_type : str, default="tukey"
            Window type to apply. Supported options are ``"tukey"`` and
            ``"hann"``.

        alpha : float, default=0.1
            Shape parameter for the Tukey window. Only used when
            ``w_type="tukey"``.

        dim : str, default="time"
            Dimension along which the tapering window is applied.

    Returns
    -------
        xr.DataArray
            Tapered DataArray with the original ``time`` and ``distance``
            coordinates.
    """

    axis = xarr.dims.index(dim)
    
    if w_type=='tukey':    
        win_shape = [1] * xarr.ndim
        win_shape[axis] = xarr.shape[axis]
        win=sps.windows.tukey(xarr.shape[axis], alpha, sym=True).reshape(win_shape)
    elif w_type=='hann':
        win_shape = [1] * xarr.ndim
        win_shape[axis] = xarr.shape[axis]
        win=sps.windows.hann(xarr.shape[axis], sym=True).reshape(win_shape)
    else:
        raise ValueError("w_type must be 'tukey' or 'hann'")

    if isinstance(xarr.data, da.Array):
        win = da.from_array(win, chunks=win.shape)
            
    xarr_tap = xr.DataArray(xarr.data * win, dims=xarr.dims, coords=xarr.coords, attrs=xarr.attrs)        
    return xarr_tap


####################################################################

def demean(
    xarr: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray:

    """
    Remove the mean along a given dimension of a DAS DataArray.

    This function is Dask-compatible and subtracts the mean value along the
    selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        dim : str, default="time"
            Dimension along which the mean is removed.

    Returns
    -------
        xr.DataArray
            Demeaned DataArray with the original ``time`` and ``distance``
            coordinates.
    """

    axis = xarr.dims.index(dim)
    
    mean = xarr.mean(axis=axis)
    xarr_demean = xarr - mean
    xarr_demean.attrs = xarr.attrs

    return xarr_demean

####################################################################

def detrend(
    xarr: xr.DataArray,
    poly: bool = False,
    degree: int = 4,
    dim: str = "time",
) -> xr.DataArray:

    """
    Remove a linear or polynomial trend from a DAS DataArray.

    This function is Dask-compatible and detrends the data along the selected
    dimension. By default, a linear detrend is applied.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        poly : bool, default=False
            If False, apply linear detrending.
            If True, apply polynomial detrending using the specified degree.

        degree : int, default=4
            Degree of the polynomial fit. Only used when ``poly=True``.

        dim : str, default="time"
            Dimension along which the detrending is applied.

    Returns
    -------
        xr.DataArray
            Detrended DataArray with the original ``time`` and ``distance``
            coordinates.

    Notes
    -----
        Polynomial detrending is recommended only for small datasets.
    """

    axis = xarr.dims.index(dim)

    def poly_det_func(block, axis):
        # can be optimised better
        def _detrend_1d(sig):
            x = np.arange(sig.shape[0])
            coeffs = np.polyfit(x, sig, degree)
            trend = np.polyval(coeffs, x)
            return sig - trend
        return np.apply_along_axis(_detrend_1d, axis, block)
    
    if poly:
        if isinstance(xarr.data, da.Array) and len(xarr.data.chunks[axis]) > 1: 
            # rechunk to a single chunk along the axis
            xarr = tasks.rechunk_along_axis(xarr, axis=axis)
        func = partial(poly_det_func, axis=axis)
    else:
        func = partial(sps.detrend, axis=axis)

    arr = xarr.data
    arr_det = arr.map_blocks(func, dtype=arr.dtype) if isinstance(arr, da.Array) else func(arr)

    return xr.DataArray(arr_det, coords=xarr.coords, dims=xarr.dims, attrs=xarr.attrs)

####################################################################

def hilbert(
    xarr: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray:

    """
    Apply a Hilbert transform to a DAS DataArray along a given dimension.

    This function is Dask-compatible and uses ``scipy.signal.hilbert`` to
    compute the analytic signal.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        dim : str, default="time"
            Dimension along which the Hilbert transform is applied.

    Returns
    -------
        xr.DataArray
            Complex-valued DataArray containing the analytic signal with the
            original ``time`` and ``distance`` coordinates.
    """
    
    axis = xarr.dims.index(dim)
    
    if isinstance(xarr.data, da.Array):
        # chunks don't work well with sps.hilbert as it's an fft-based computation
        # -> rechunk to a single chunk along the axis
        if len(xarr.data.chunks[axis]) > 1:
            xarr=tasks.rechunk_along_axis(xarr, axis=axis)

        arr = xarr.data
        arr_hil = arr.map_blocks(partial(sps.hilbert, axis=axis), dtype=arr.dtype)

    else:
        arr_hil = sps.hilbert(xarr.data, axis=axis)

    return xr.DataArray(arr_hil, coords=xarr.coords, dims=xarr.dims, attrs=xarr.attrs)


####################################################################

def filter(
    xarr: xr.DataArray,
    fcut: float | tuple[float, float] | None = None,
    filt_type: str = "lowpass",
    order: int = 4,
    dim: str = "time",
    zerophase: bool = True,
) -> xr.DataArray:

    """
    Apply a Butterworth filter to a DAS DataArray along a given dimension.

    This function is Dask-compatible and uses the SciPy Butterworth filter
    design (``scipy.signal.butter``) with second-order sections (SOS)
    representation (``scipy.signal.sosfilt`` / ``scipy.signal.sosfiltfilt``).

    Filtering is applied along the selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        fcut : float or tuple[float, float], default=None
            Cutoff frequency of the filter. A scalar value is used for low-pass
            and high-pass filters. A ``(low, high)`` tuple is used for band-pass
            and band-stop filters.

        filt_type : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
            Type of Butterworth filter.

        order : int, default=4
            Order of the Butterworth filter.

        dim : str, default="time"
            Dimension along which the filtering is applied.

        zerophase : bool, default=True
            If True, apply zero-phase filtering using ``scipy.signal.sosfiltfilt``.
            For Dask arrays, zero-phase filtering requires the full signal along
            the filtering dimension and may require rechunking.

    Returns
    -------
        xr.DataArray
            Filtered DataArray with the original ``time`` and ``distance``
            coordinates.
    """

    if fcut is None:
        raise ValueError("The cutoff frequency (fcut) must be defined either as (scalar) "
                         "or a tuple for bandpass/bandstop (low, high)")
    
    if dim == 'time':
        fs=1/xarr.attrs['dt']
    elif dim == 'distance':
        fs=1/xarr.attrs['dx']

    axis = xarr.dims.index(dim)
    
    nyquist = 0.5 * fs
    # normalize fcut on nyquist
    fcut = np.array(fcut) / nyquist

    # define the filter
    valid = {"lowpass", "highpass", "bandpass", "bandstop"}
    if filt_type not in valid:
        raise ValueError(
            f"filt_type must be one of {valid}"
        )
    sos = sps.butter(order, fcut, btype=filt_type, output="sos")
    
    # wrapper function to be used with lazy arrays
    def filt_block(block):
        return sps.sosfilt(sos, block, axis)
    def filtfilt_block(block):
        return sps.sosfiltfilt(sos, block, axis)
    
    if isinstance(xarr.data, da.Array):
        # rechunk to a single chunk along 'dim'
        if len(xarr.data.chunks[axis]) > 1:
            xarr=tasks.rechunk_along_axis(xarr, axis=axis)
        arr = xarr.data
        
        if zerophase:
            arr_filt = arr.map_blocks(filtfilt_block, dtype=arr.dtype)
        else:
            arr_filt = arr.map_blocks(filt_block, dtype=arr.dtype)
    else:
        # apply the SOS filter to the data
        if zerophase:
            arr_filt = sps.sosfiltfilt(sos, xarr.data, axis) 
            # forward-backward filtering, also better for short signals -> reduce edge effects
        else:
            arr_filt = sps.sosfilt(sos, xarr.data, axis)

    return xr.DataArray(arr_filt, dims=xarr.dims, coords=xarr.coords, attrs=xarr.attrs)


####################################################################

def fft(
    xarr: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray:

    """
    Compute a Fourier transform of a DAS DataArray along a given dimension.

    This function is Dask-compatible and computes the spectrum along the
    selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        dim : str, default="time"
            Dimension along which the Fourier transform is applied.

    Returns
    -------
        xr.DataArray
            Complex-valued DataArray containing the Fourier spectrum.

            If ``dim="time"``, the output coordinates are ``frequency`` and
            ``distance``.

            If ``dim="distance"``, the output coordinates are ``time`` and
            ``wavenumber``.
    """

    axis = xarr.dims.index(dim)
    
    new_dim = None
    frequency = None
    #if axis==time_axis:
    if dim == 'time':
        new_dim = 'frequency'
        # `time` frequency
        coord_value = np.fft.fftshift(np.fft.fftfreq(xarr.sizes['time'],xarr.attrs['dt']))
    elif dim == 'distance':
        new_dim = 'wavenumber'
        # `space` frequency
        coord_value = np.fft.fftshift(np.fft.fftfreq(xarr.sizes['distance'],xarr.attrs['dx']))
        
    if isinstance(xarr.data, da.Array):
        
        if len(xarr.data.chunks[axis]) > 1:
            xarr=tasks.rechunk_along_axis(xarr, axis=axis)
        arr = xarr.data

        def fft_block(block, axis):
            return np.fft.fftshift(np.fft.fft(block, axis=axis), axes=axis)

        arr_fft = arr.map_blocks(partial(fft_block, axis=axis), dtype=np.complex64)
    else:
        arr_fft = np.fft.fftshift(np.fft.fft(xarr.data, axis=axis), axes=axis)

    # Replace dim name, add frequency coordinate and updating metadata
    old_dim = xarr.dims[axis]
    dims = list(xarr.dims)
    dims[axis] = new_dim

    coords = OrderedDict()
    for d in dims:
        if d == new_dim:
            coords[new_dim] = coord_value
        else:
            coords[d] = xarr.coords[d]

    xarr_fft = xr.DataArray(arr_fft, dims=dims, coords=coords)
    xarr_fft.attrs = copy.deepcopy(xarr.attrs) # safer when modifying attrs
    xarr_fft.attrs['dimensionsU'][axis] += '-1'
    xarr_fft.attrs['dimensions'][axis] = new_dim
    
    # in case of DataArray.sel() step, save new cut time0 and distance0 for ifft
    # add in metadata of the original reference at 0 for time or distance
    if dim == 'time':
        xarr_fft.attrs['time0'] = xarr.coords['time'][0].values
    elif dim == 'distance':
        xarr_fft.attrs['distance0'] = xarr.coords['distance'][0].values
   
    return xarr_fft

####################################################################

def ifft(
    xarr: xr.DataArray,
    dim: str = "frequency",
) -> xr.DataArray:

    """
    Compute the inverse Fourier transform of a DAS DataArray.

    This function is Dask-compatible and reconstructs the signal from its
    Fourier spectrum along the selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input complex-valued DAS spectrum DataArray. Expected dimensions are
            ``frequency`` and ``distance`` or ``time`` and ``wavenumber``.

        dim : str, default="frequency"
            Dimension along which the inverse Fourier transform is applied.

    Returns
    -------
        xr.DataArray
            Reconstructed DAS DataArray with ``time`` and ``distance`` coordinates.
    """
    
    axis = xarr.dims.index(dim)
    
    new_dim = None
    frequency = None
    #if axis==time_axis:
    if dim == 'frequency':
        new_dim = 'time'
        c = np.linspace(0, xarr.shape[axis]*xarr.attrs['dt'], xarr.shape[axis], endpoint=False)
        coord_value = xarr.attrs['time0'] + np.array(c * 1e9, dtype='timedelta64[ns]')

    #elif axis==dist_axis:
    elif dim == 'wavenumber':
        new_dim = 'distance'
        c = np.linspace(0, xarr.shape[axis]*xarr.attrs['dx'], xarr.shape[axis], endpoint=False)
        coord_value = xarr.attrs['distance0'] + np.array(c)
        
    if isinstance(xarr.data, da.Array):
        
        if len(xarr.data.chunks[axis]) > 1:
            xarr=tasks.rechunk_along_axis(xarr, axis=axis)
        arr = xarr.data

        def ifft_block(block, axis):
            return np.fft.ifft(np.fft.ifftshift(block, axes=axis), axis=axis)
                    
        arr_ifft = arr.map_blocks(partial(ifft_block, axis=axis), dtype=np.complex64)
    else:
        arr_ifft = np.fft.ifft(np.fft.ifftshift(xarr.data, axes=axis), axis=axis)

    # Replace dim name, add frequency coordinate and updating metadata
    old_dim = xarr.dims[axis]
    dims = list(xarr.dims)
    dims[axis] = new_dim

    coords = OrderedDict()
    for d in dims:
        if d == new_dim:
            coords[new_dim] = coord_value
        else:
            coords[d] = xarr.coords[d]

    xarr_ifft = xr.DataArray(arr_ifft, dims=dims, coords=coords)
    xarr_ifft.attrs = copy.deepcopy(xarr.attrs) # safer when modifying attrs
    xarr_ifft.attrs['dimensionsU'][axis] = xarr_ifft.attrs['dimensionsU'][axis].removesuffix('-1')
    xarr_ifft.attrs['dimensions'][axis] = new_dim
            # add in metadata of the original reference at 0 for time or distance
    if dim == 'frequency':
        del xarr_ifft.attrs['time0']
    elif dim == 'wavenumber':
        del xarr_ifft.attrs['distance0']
   
    return xarr_ifft

####################################################################

def fk(
    xarr: xr.DataArray,
) -> xr.DataArray:

    """
    Compute the frequency-wavenumber (FK) transform of a DAS DataArray.

    This function is Dask-compatible and computes the 2D Fourier transform
    along the ``time`` and ``distance`` dimensions.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

    Returns
    -------
        xr.DataArray
            Complex-valued FK spectrum with ``frequency`` and ``wavenumber``
            coordinates.
    """
    
    time_axis = xarr.dims.index('time')
    dist_axis = xarr.dims.index('distance')

    if isinstance(xarr.data, da.Array):
        # if chunk is 1 or smallest in one dimension
        # performs fft first on it (avoid rechunking twice)
        time_chunks = len(xarr.data.chunks[time_axis])
        dist_chunks = len(xarr.data.chunks[dist_axis])

        if time_chunks <= dist_chunks:
            dim_1 = 'time'
            dim_2 = 'distance'
        else:
            dim_1 = 'distance'
            dim_2 = 'time'
    
        xarr_fk = fft(xarr, dim=dim_1)
        xarr_fk = fft(xarr_fk, dim=dim_2)

    else:
        xarr_fk = fft(xarr, dim='time')
        xarr_fk = fft(xarr_fk, dim='distance')
   
    return xarr_fk

####################################################################

def ifk(
    xarr: xr.DataArray,
) -> xr.DataArray:

    """
    Compute the inverse frequency-wavenumber (FK) transform of a DAS DataArray.

    This function is Dask-compatible and reconstructs the DAS signal from
    its FK spectrum.

    Parameters
    ----------
        xarr : xr.DataArray
            Input complex-valued FK spectrum with ``frequency`` and ``wavenumber``
            coordinates.

    Returns
    -------
        xr.DataArray
            Reconstructed DAS DataArray with ``time`` and ``distance``
            dimensions.
    """
    
    freq_axis = xarr.dims.index('frequency')
    wavenb_axis = xarr.dims.index('wavenumber')

    if isinstance(xarr.data, da.Array):
        # if chunk is 1 or smallest in one dimension
        # performs ifft first on it (avoid rechunking twice)
        freq_chunks = len(xarr.data.chunks[freq_axis])
        wavenb_chunks = len(xarr.data.chunks[wavenb_axis])
            
        if freq_chunks <= wavenb_chunks:
            dim_1 = 'frequency'
            dim_2 = 'wavenumber'
        else:
            dim_1 = 'wavenumber'
            dim_2 = 'frequency'
    
        xarr_ifk = ifft(xarr, dim=dim_1)
        xarr_ifk = ifft(xarr_ifk, dim=dim_2)
    else:
        xarr_ifk = ifft(xarr, dim='frequency')
        xarr_ifk = ifft(xarr_ifk, dim='wavenumber')
   
    return xarr_ifk

####################################################################

def _build_fk_filter(C, ccut, taper, side):
    """
    Create a smooth FK filter mask based on velocity fan limits.
        Parameters: see fk_filt for parameters description
        Returns: fkfilt (ndarray): 2D array of same shape as `C`
    """

    def apply_log_filter(log_C):
        # apply passband
        passband = (log_C >= log_ca_min) & (log_C <= log_ca_max)
        fkfilt[passband] = 1.0

        # tapering lower side
        taper_min = (log_C >= log_ca_min_ext) & (log_C < log_ca_min)
        fkfilt[taper_min] = 0.5 * (1 + np.cos(np.pi * (log_C[taper_min] - log_ca_min) / taper))

        # tapering upper side
        taper_max = (log_C > log_ca_max) & (log_C <= log_ca_max_ext)
        fkfilt[taper_max] = 0.5 * (1 + np.cos(np.pi * (log_C[taper_max] - log_ca_max) / taper))

    # initialize filter mask
    fkfilt = np.zeros_like(C, dtype=float)

    # convert velocity bounds to logarithmic scale
    if ccut[0] is None:
        log_ca_min = -np.inf # No lower limit (use full range)
    else:
        log_ca_min = np.log(ccut[0])
    if ccut[1] is None:
        log_ca_max = np.inf  # No upper limit (use full range)
    else:
        log_ca_max = np.log(ccut[1])

    # tapering bounds
    log_ca_min_ext = log_ca_min - taper
    log_ca_max_ext = log_ca_max + taper

    # apply filter to positive or negative C as requested
    if side in ('pos', 'both'):
        log_C_pos = np.full_like(C, np.nan, dtype=float)
        eps = np.finfo(float).tiny  # to avoid log(0)
        log_C_pos[C > 0] = np.log(np.maximum(C[C > 0], eps))
        apply_log_filter(log_C_pos)
    
    if side in ('neg', 'both'):
        log_C_neg = np.full_like(C, np.nan, dtype=float)
        eps = np.finfo(float).tiny  # to avoid log(0) 
        log_C_neg[C < 0] = np.log(np.maximum(-C[C < 0], eps))
        apply_log_filter(log_C_neg)

    # apply symmetric vertical taper around center (k=0) to limit artefacts
    if side in ('neg', 'pos'):
        ncols = C.shape[1]
        center = ncols // 2
        taper_width = max(1, ncols // 32)

        dist = np.abs(np.arange(ncols) - center)
        taper = np.ones(ncols)
        mask = dist < taper_width # define taper zone

        taper[mask] = 0.5 * (1 - np.cos(np.pi * dist[mask] / taper_width))
        fkfilt *= taper

    return fkfilt
    

def fk_filt(
    xarr: xr.DataArray,
    ccut: tuple[float | None, float | None] = (None, None),
    taper: float = 0.5,
    side: str = "both",
) -> xr.DataArray:

    """
    Apply a velocity filter in the FK domain.

    This function is Dask-compatible and filters an FK-transformed DAS
    DataArray using velocity limits. The tapering applied in log scale 
    in order to get similar taper width when limits of the velocity fan
    are of different orders of magnitude.

    Parameters
    ----------
        xarr : xr.DataArray
            Input FK DataArray with ``frequency`` and ``wavenumber`` dimensions.

        ccut : tuple[float | None, float | None], default=(None, None)
            Velocity cutoff limits defining the filter range.

            - ``(vel1, vel2)``: band-pass velocity filter.
            - ``(vel1, None)``: high-pass velocity filter.
            - ``(None, vel2)``: low-pass velocity filter.

        taper : float, default=0.5
            Width of the velocity taper in logarithmic scale.

        side : {"both", "neg", "pos"}, default="both"
            Select the wavenumber components to filter.

            - ``"both"``: filter positive and negative wavenumbers.
            - ``"neg"``: keep and filter negative wavenumbers only.
            - ``"pos"``: keep and filter positive wavenumbers only.

    Returns
    -------
        xr.DataArray
            Filtered FK DataArray with ``frequency`` and ``wavenumber``
            coordinates.
    """

    arr = xarr.data
    frequency = xarr.coords['frequency']
    k = xarr.coords['wavenumber']

    freq_dim = xarr.get_axis_num('frequency')
    k_dim = xarr.get_axis_num('wavenumber')

    if isinstance(arr, da.Array):
        # the goal is to get a 2D array (fk dimensions) of the velocity C = F/K
        # but building it so that C remains chunked from start (no meshgrid to get F and K)
        # using meshgrid forces a single worker to materialize the full array C at once
        
        freq_chunks = arr.chunks[freq_dim]
        k_chunks = arr.chunks[k_dim]

        # Since arr is complex, finds the dtype used only for the real 
        # component of arr to avoid double memory
        real_dtype = np.zeros(1, dtype=arr.dtype).real.dtype
        
        F = da.from_array(frequency.values.astype(real_dtype), chunks=freq_chunks)
        K = da.from_array(k.values.astype(real_dtype), chunks=k_chunks)

        F = F[:, None]  # (n_freq, 1)
        K = K[None, :]  # (1, n_k)
        K = da.where(K == 0, np.nan, K)  # avoid division by zero warning

        C = F / K  # create C from K and F leading to C being chunked as arr

        def wraped_build_fk_filter(C_block):
            return _build_fk_filter(C_block, ccut, taper, side)

        fkfilt = C.map_blocks(wraped_build_fk_filter, dtype=arr.dtype)
        # ensures that fkfilt has chunking compatible with arr
        fkfilt = fkfilt.rechunk(arr.chunks)

    else:
        # Define velocity grid
        K,F = np.meshgrid(k,frequency)
        K = np.where(K == 0, np.nan, K)  # avoid division by zero warning
        C = F / K
        
        fkfilt = _build_fk_filter(C, ccut, taper, side)

    fft2 = arr*fkfilt

    xarr_fkfilt = xr.DataArray(fft2, dims=xarr.dims, coords=xarr.coords, attrs=xarr.attrs)
    
    return xarr_fkfilt
    
####################################################################

def rms(
    xarr: xr.DataArray,
    win: float = 1,
    dim: str = "time",
) -> xr.DataArray:
    """
    Compute the root mean square (RMS) over a sliding window.

    This function is Dask-compatible and applies a rolling RMS calculation
    along the selected dimension.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        win : float, default=1
            Sliding window length in seconds.

        dim : str, default="time"
            Dimension along which the rolling RMS is computed.

    Returns
    -------
        xr.DataArray
            RMS DataArray with the same ``time`` and ``distance`` coordinates
            and preserved attributes.
    """

    samples = int(win/xarr.attrs['dt']) # convert seconds to nb samples
    rms = (xarr ** 2).rolling({dim: samples}, center=True, min_periods=1).mean() ** 0.5
    rms.attrs = xarr.attrs.copy()
    
    return rms
    
####################################################################

def psd(
    xarr: xr.DataArray,
    win: float | None = None,
) -> xr.DataArray:

    """
    Compute the power spectral density (PSD) of a DAS DataArray.

    This function is Dask-compatible and uses ``scipy.signal.welch`` to
    estimate the PSD along the time dimension independently for each
    distance channel.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        win : float, default=None
            Window length used for the Welch PSD calculation. If ``None``, the
            full time series is used.

    Returns
    -------
        xr.DataArray
            PSD DataArray with ``frequency`` and ``distance`` coordinates.
    """

    fs = 1/xarr.attrs['dt']
    
    if win is None:
        win = xarr.sizes['time']

    # welch function wrapper
    def _welch(block, fs, win):
        frequency, spec = sps.welch(block, fs, nperseg=win, axis=-1)
        return spec

    if isinstance(xarr.data, da.Array):
        # rechunk along time
        axis = xarr.dims.index('time')
        xarr = tasks.rechunk_along_axis(xarr, axis=axis)

    # define the new frequency coord 
    nperseg = win if win is not None else len(xarr.time)
    frequency = np.fft.rfftfreq(nperseg, d=1/fs)  # defines the frequency associated with fft window

    # apply welch wrapper on the data
    psd_array = xr.apply_ufunc(
        _welch,
        xarr,
        kwargs=dict(fs=fs, win=win),
        input_core_dims=[['time']],
        output_core_dims=[['frequency']],
        dask='parallelized',               
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(
        output_sizes={'frequency': len(frequency)}
        ),
    )

    psd_array = psd_array.assign_coords(frequency=frequency)
    psd_array.attrs['name'] = "psd"
    psd_array.attrs['unit'] = f"({xarr.attrs['unit']})² Hz⁻¹" # default scaling "density"

    return psd_array

####################################################################

def spectrograms(
    xarr: xr.DataArray,
    win: float | None = None,
    overlap: float = 0.5,
) -> xr.DataArray:
    """
    Compute spectrograms for each channel of a DAS DataArray.

    This function is Dask-compatible and uses ``scipy.signal.spectrogram`` to
    compute the time-frequency representation independently for each distance
    channel.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.
            ``distance`` represents the selected channels.

        win : float, default=None
            Window length used for the spectrogram calculation. If ``None``,
            the default window length from ``scipy.signal.spectrogram`` is used.

        overlap : float, default=0.5
            Fraction of the window length used as overlap between consecutive
            segments.

    Returns
    -------
        xr.DataArray
            Spectrogram DataArray with ``distance``, ``frequency`` and ``time``
            coordinates.

    Notes
    -----
        For efficient processing, it is recommended to use a coarser channel
        spacing before computing spectrograms, for example with
        ``daspal.process.select_channels``.

        The output can be visualized using ``daspal.plot.spectrogram`` after
        selecting individual channels or distances.
    """
    
    sig = xarr.copy()
    
    fs=1/sig.attrs['dt']
    start = pd.to_datetime(sig.coords['time'][0].values)

    # spectrogram function wrapper
    def _spectrogram(block, fs, win, overlap):

        noverlap = None if win is None else int(overlap * win)
        
        frequency, t, spec = sps.spectrogram(block, fs=fs, window=('tukey', 0.1),
                                              nperseg=win, noverlap=noverlap, nfft=None,
                                              detrend='constant', return_onesided=True,
                                              scaling='density', axis=-1,
                                              mode='magnitude')
        return spec

    if isinstance(xarr.data, da.Array):
        # rechunk along time
        xarr = xarr.chunk({"time": -1})

    # define the new frequency and time coord
    nperseg = win if win is not None else xarr.sizes["time"]
    noverlap = int(overlap * nperseg)
    frequency = np.fft.rfftfreq(nperseg, d=1/fs) # defines the frequency associated with fft window
    step = nperseg - noverlap
    n_windows = 1 + (xarr.sizes["time"] - nperseg) // step # defines the number of time samples
    t = np.arange(n_windows) * step / fs + nperseg/(2*fs) # get center times in seconds from samples
    time = start + pd.to_timedelta(t, unit="s")

    # apply welch wrapper on the data
    spectro = xr.apply_ufunc(
        _spectrogram,
        xarr,
        kwargs=dict(fs=fs, win=win, overlap=overlap),
        input_core_dims=[['time']],
        output_core_dims=[['frequency','time']],
        exclude_dims={'time'}, # exclude original time dimension
        dask='parallelized',               
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(
        output_sizes={'frequency': len(frequency), 'time': len(time)}
        ),
    )

    spectro = spectro.assign_coords(frequency=frequency, time=time)
    spectro.attrs['name'] = "spectrograms"
    spectro.attrs['unit'] = f"({sig.attrs['unit']})² Hz⁻¹"  # default scaling "density"
    
    return spectro