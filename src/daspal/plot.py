"""
Library for visualisation of DAS data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import pandas as pd
import xarray as xr

import dask.array as dask_array

import warnings

import daspal.signal as signal

import holoviews as hv
import holoviews.operation.datashader as hd

###############################################################################

def _font_scaling(ax, panel_size=4, fontsize=12):
    """
    Font size defined by a figure subplot (panel) size read from ax.
    panel_size : reference subplot size in inches
    fontsize  : font size at that panel size
    """
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()

    # axis size as fraction of figure
    bbox = ax.get_position()
    panel_w = bbox.width * fig_w
    panel_h = bbox.height * fig_h

    # geometric mean balances width and height
    scale = (panel_w * panel_h)**0.5 / panel_size
    
    return fontsize * scale

def _apply_fonts(ax, panel_size=4, fontsize=12, cbar=None):
    """
    Apply consistent typography to an axis and optional colorbar.
    """

    font = _font_scaling(ax, panel_size, fontsize)
    
    # titles and labels fonts
    if ax.title:
        ax.title.set_fontsize(1.2 * font)

    ax.xaxis.label.set_fontsize(1.2 * font)
    ax.yaxis.label.set_fontsize(1.2 * font)

    # ticks fonts
    ax.tick_params(axis='both', which='major', labelsize=font)
    ax.tick_params(axis='both', which='minor', labelsize=0.8 * font)

    # offset text fonts (dates, scientific notation)
    ax.xaxis.get_offset_text().set_fontsize(font)
    ax.yaxis.get_offset_text().set_fontsize(font)

    # legend fonts
    leg = ax.get_legend()
    if leg:
        for txt in leg.get_texts():
            txt.set_fontsize(font)
        if leg.get_title():
            leg.get_title().set_fontsize(1.1 * font)

    # colorbar fonts
    if cbar is not None:
        cbar.ax.tick_params(labelsize=font)

        if cbar.orientation == "vertical":
            cbar.ax.yaxis.label.set_fontsize(1.2 * font)
            cbar.ax.yaxis.get_offset_text().set_fontsize(font)
        else:
            cbar.ax.xaxis.label.set_fontsize(1.2 * font)
            cbar.ax.xaxis.get_offset_text().set_fontsize(font)

###############################################################################

def das(
    xarr: xr.DataArray,
    ax=None,
    cmin: float | None = None,
    cmax: float | None = None,
    cbar: bool = True,
    log: bool = False,
    d_unit: str = "m",
    interactive: bool = False,
    hover: bool = True,
    width: float = 4,
    height: float = 4,
) -> object:

    """
    Plot DAS data as a time-distance image.

    This function is Dask-compatible and can accept lazy DataArrays, making
    it suitable for visualization of large DAS datasets.

    The output can be generated either as a Matplotlib plot or as an
    interactive Holoviews object.

    Parameters
    ----------
        xarr : xr.DataArray
            Input DAS DataArray with ``time`` and ``distance`` dimensions.

        ax : matplotlib.axes.Axes, default=None
            Matplotlib axis where the figure is drawn. If ``None``, a new axis
            is created.

        cmin : float, default=None
            Minimum value for the color scale.

        cmax : float, default=None
            Maximum value for the color scale.

        cbar : bool, default=True
            If True, display the colorbar.

        log : bool, default=False
            If True, compute the envelope over 'time' and convert amplitudes to decibels (dB).

        d_unit : {"m", "km"}, default="m"
            Unit used for the distance axis.

        interactive : bool, default=False
            If True, return an interactive Holoviews plot instead of a Matplotlib
            plot.

        hover : bool, default=True
            Enable interactive value display when hovering over the plot.

        width : float, default=4
            Figure width in inches.

        height : float, default=4
            Figure height in inches.

    Returns
    -------
        object
            Matplotlib axis if ``interactive=False``.
            Holoviews object if ``interactive=True``.
    """

    # --- preprocessing ---
    start = pd.to_datetime(xarr.coords['time'].data[0])
    time_window = (xarr.coords['time'].data[-1] - xarr.coords['time'].data[0]) / np.timedelta64(1, 's')

    if d_unit == 'km':
        xarr = xarr.assign_coords(distance=xarr.distance/1000)
    unit = xarr.attrs.get('unit', '')

    # --- compute enveloppe if log ---
    if log:
        eps = np.finfo(np.float32).tiny   # very small float ~1.2e-38, to avoid log(0)
        sig_env = signal.hilbert(xarr, dim='time')
        da = 10 * np.log10(np.maximum(np.abs(sig_env), eps))
    else:
        da = xarr

    cmap = 'viridis' if log else 'RdBu'
    
    # --- holoviews interactive plot ---
    if interactive:
        if isinstance(da.data, dask_array.Array):
        # persist data for better interactive dynamics
            da = da.persist()

        cmax = cmax if cmax is not None else (
            float(da.max().compute()) if log else float(np.abs(da).max().compute()/5) # dividing by 5 to see more saturation
        )
        cmin = cmin if cmin is not None else (
            cmax - 30 if log else -cmax
        )
        
        dpi=100 # default 100 dpi
        width_px  = int(width * dpi)
        height_px = int(height * dpi)

        image = hv.Image(da, kdims=['time', 'distance'])#, label='DAS Time vs Distance')
        opts = {
            **({'clim': (cmin, cmax)}),# define clim once to speed up the rendering
            'cmap': cmap,
            'colorbar': cbar,
            'xlabel': 'Time',
            'ylabel': f'Distance ({d_unit})',
            'width': width_px,
            'height': height_px,
            'invert_yaxis': True,
            'clabel': f"{'dB rel. 1 ' if log else ''}{unit}",
            'toolbar': 'above',
            'colorbar_opts': {'height': height_px // 2, 'width': width_px // 40}
        }
        if hover:
            opts['tools'] = ['hover']
        return hd.rasterize(image).opts(**opts)
    
    elif not interactive and isinstance(da.data, dask_array.Array):
        da = da.compute()
    
    # --- matplotlib plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    cmax = cmax if cmax is not None else (
        float(da.max()) if log else float(np.abs(da).max()/5) # dividing by 5 to see more saturation
    )
    cmin = cmin if cmin is not None else (
        cmax - 30 if log else -cmax
    )


    dist = da.distance.data
    t0 = 0 if time_window <= 120 else start
    deltat = time_window if time_window <= 120 else pd.to_timedelta(time_window, unit='s')
    da = da.transpose('distance', 'time')
    extent = [t0, t0 + deltat, dist[-1], dist[0]]

    img = ax.imshow(da.data, vmin=cmin, vmax=cmax, cmap=cmap,
                    extent=extent, aspect='auto', interpolation='none')
    
    # x-axis formatting for long times
    if time_window > 120:
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        if start.microsecond == 0:
            ax.set_xlabel(f"Time (s) ref. {start.strftime('%d/%m/%Y %H:%M:%S')} UTC")
        else:
            ax.set_xlabel(f"Time (s) ref. {start.strftime('%d/%m/%Y %H:%M:%S')}.{start.microsecond//1000:03d} UTC")

    cb = None
    if cbar:
        cb = plt.colorbar(img, ax=ax, extend="both", shrink=0.5, pad=0.025, aspect=15)
        cb.set_label(f"{'dB rel. 1 ' if log else ''}{unit}")

    ax.set_ylabel(f"Distance ({d_unit})")

    # scale fonts to subplot panel size
    _apply_fonts(ax, cbar=cb)

    return ax
    
####################################################################

def psd(
    xarr: xr.DataArray,
    ax=None,
    cmin: float | None = None,
    cmax: float | None = None,
    cbar: bool = True,
    log_freqaxis: bool = False,
    d_unit: str = "m",
    interactive: bool = False,
    width: float = 4,
    height: float = 4,
) -> object:

    """
    Plot the power spectral density (PSD) as a frequency-distance image.

    This function is intended for PSD DataArrays produced by
    ``daspal.signal.psd``. It can generate either a Matplotlib plot or an
    interactive Holoviews plot.

    Parameters
    ----------
        xarr : xr.DataArray
            Input PSD DataArray with ``frequency`` and ``distance`` dimensions.

        ax : matplotlib.axes.Axes, default=None
            Matplotlib axis where the figure is drawn. If ``None``, a new axis
            is created.

        cmin : float, default=None
            Minimum value for the color scale.

        cmax : float, default=None
            Maximum value for the color scale.

        cbar : bool, default=True
            If True, display the colorbar.

        log_freqaxis : bool, default=False
            If True, display the frequency axis using a logarithmic scale.

        d_unit : {"m", "km"}, default="m"
            Unit used for the distance axis.

        interactive : bool, default=False
            If True, return an interactive Holoviews plot.

        width : float, default=4
            Figure width in inches.

        height : float, default=4
            Figure height in inches.

    Returns
    -------
        object
            Matplotlib axis if ``interactive=False``.
            Holoviews object if ``interactive=True``.
    """

    psd_da = xarr.copy()

    # --- preprocessing ---
    if d_unit == 'km':
        psd_da = psd_da.assign_coords(distance=psd_da.distance/1000)
    
    # remove zero frequency if present
    psd_da = psd_da.isel(frequency=slice(1, None))

    eps = np.finfo(float).tiny   # very small float ~2e-308, to avoid log(0)
    psd = 10 * np.log10(np.maximum(psd_da, eps))

    cmax = cmax if cmax is not None else np.amax(psd.data) # da.max()
    cmin = cmin if cmin is not None else cmax - 50

    # --- holoviews interactive plot ---
    if interactive:
        dpi=100 # default 100 dpi value from matplotlib for comp
        width_px  = int(width * dpi)
        height_px = int(height * dpi)
        if log_freqaxis==True:
            psd = psd.assign_coords(frequency=np.log10(psd.frequency))
        # Use QuadMesh for log frequency or irregular grids
        image = hv.QuadMesh(psd, kdims=['frequency','distance'])#, label='DAS frequency vs Distance')

        return hd.rasterize(image).opts(
            clim=(cmin, cmax),
            cmap='cividis',
            colorbar=cbar,
            #title='DAS frequency vs Distance',
            tools=['hover'],
            xlabel='Frequency (Hz)',
            ylabel=f'Distance ({d_unit})',
            width=width_px,
            height=height_px,
            invert_yaxis=True,
            clabel=f'PSD dB rel. 1 {psd_da.attrs["unit"]}',
            toolbar='above',
            colorbar_opts={'height':height_px//2, 'width':width_px//20}
        )

    # --- matplotlib plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    freq = psd.frequency.data
    dist = psd.distance.data
    psd = psd.transpose('distance', 'frequency')
    extent = [freq[0], freq[-1], dist[-1], dist[0]]

    img = ax.imshow(psd.data, vmin=cmin, vmax=cmax, cmap='cividis',
                    extent=extent, aspect='auto', interpolation='none')

    cb = None
    if cbar:
        cb = plt.colorbar(img, ax=ax, extend="both", shrink=0.5, pad=0.025, aspect=15)
        cb.set_label(f'PSD dB rel. 1 {psd_da.attrs["unit"]}')
    
    if log_freqaxis:
        ax.set_xscale('log')
    
    ax.set_xlim(0.01 if freq[0] < 0.01 else freq[0], freq[-1])
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'Distance ({d_unit})')
    
    # scale fonts to subplot panel size
    _apply_fonts(ax, cbar=cb)
    
    return ax

####################################################################

def spectrogram(
    xarr: xr.DataArray,
    ax=None,
    cmin: float | None = None,
    cmax: float | None = None,
    cbar: bool = True,
    log_freqaxis: bool = False,
    cbar_orient: str = "vertical",
    interactive: bool = False,
    width: float = 8,
    height: float = 3,
) -> object:

    """
    Plot a single-channel DAS spectrogram.

    This function is intended for spectrogram DataArrays produced by
    ``daspal.signal.spectrograms``. It can generate either a Matplotlib plot or an
    interactive Holoviews plot.

    Parameters
    ----------
        xarr : xr.DataArray
            Input spectrogram DataArray with ``time`` and ``frequency``
            dimensions.

        ax : matplotlib.axes.Axes, default=None
            Matplotlib axis where the figure is drawn. If ``None``, a new axis
            is created.

        cmin : float, default=None
            Minimum value for the color scale.

        cmax : float, default=None
            Maximum value for the color scale.

        cbar : bool, default=True
            If True, display the colorbar.

        log_freqaxis : bool, default=True
            If True, display the frequency axis using a logarithmic scale.

        cbar_orient : {"horizontal", "vertical"}, default="vertical"
            Orientation of the colorbar.

        interactive : bool, default=False
            If True, return an interactive Holoviews plot.

        width : float, default=8
            Figure width in inches.

        height : float, default=3
            Figure height in inches.

    Returns
    -------
        object
            Matplotlib axis if ``interactive=False``.
            Holoviews object if ``interactive=True``.
    """

    spec_da = xarr.copy()

    # --- preprocessing ---
    # remove zero frequency if present
    spec_da = spec_da.isel(frequency=slice(1, None))

    eps = np.finfo(float).tiny   # very small float ~2e-308, to avoid log(0)
    spec = 10 * np.log10(np.maximum(spec_da, eps))
    
    cmax = cmax if cmax is not None else np.nanmax(spec.data)# with -10 gives better rep
    cmin = cmin if cmin is not None else cmax-30
    
    cmap = cm.viridis.copy()
    cmap.set_bad(color='darkgrey')  # NaNs will appear grey

    if "distance" in spec_da.coords:
        dist = spec_da.coords["distance"].values
        if dist.size == 1:
            distance = dist.item()
        else:
            raise ValueError(
            "A single distance must be selected (e.g., spectro.sel(distance=...))."
        )
    elif "distance" in spec_da.attrs:
        distance = spec_da.attrs["distance"]
    else:
        raise ValueError(
            "Distance is not available. Select a single distance coordinate "
            "(e.g., spectro.sel(distance=...)) or specify it in "
            "spec_da.attrs['distance']."
        )

    title = f"Channel at distance {distance:.0f} m"

    # --- holoviews interactive plot ---
    if interactive:
        dpi=100 # default 100 dpi value from matplotlib for comp
        width_px  = int(width * dpi)
        height_px = int(height * dpi)

        if log_freqaxis==True:
            spec = spec.assign_coords(frequency=np.log10(spec.frequency))

        # Use QuadMesh for log frequency or irregular grids
        image = hv.QuadMesh(spec, kdims=['time','frequency'], label=title)
        return hd.rasterize(image).opts(
            clim=(cmin, cmax),
            cmap='viridis',
            colorbar=cbar,
            #title=title,
            tools=['hover'],
            xlabel='Time',
            ylabel='Frequency (Hz)',
            width=width_px,
            height=height_px,
            #invert_yaxis=True,
            clabel=f'dB rel. 1 {spec_da.attrs["unit"]}',
            toolbar='above',
            colorbar_opts={'height':height_px//2, 'width':width_px//40}
        )

    # --- matplotlib plotting ---
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    freq = spec.frequency.values
    time = pd.to_datetime(spec.time.values)
    extent = [time[0], time[-1], freq[-1], freq[0]]
    img=ax.imshow(spec.data, cmap=cmap, vmin=cmin, vmax=cmax,
                    extent=extent, aspect='auto', interpolation='none')

    cb = None
    if cbar:
        shrink, pad, aspect = (0.75, 0.025, 15) if cbar_orient == 'vertical' else (0.5, 0.15, 20)
        cb = plt.colorbar(img, ax=ax, extend="both", orientation=cbar_orient, shrink=shrink, pad=pad, aspect=aspect)
        cb.set_label(f'dB rel. 1 {spec_da.attrs["unit"]}')

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if log_freqaxis==True:
        ax.set_yscale("log")

    ax.set_ylim(0.01 if freq[0] < 0.01 else freq[0], freq[-1])
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    #ax.grid()

    # scale fonts to subplot panel size
    _apply_fonts(ax, cbar=cb)

    return ax

####################################################################

def fk(
    xarr: xr.DataArray,
    ax=None,
    cmin: float | None = None,
    cmax: float | None = None,
    cbar: bool = True,
    interactive: bool = False,
    width: float = 4,
    height: float = 4,
    log_axes: bool = False,
    log_xmin: float | None = None,
    log_ymin: float | None = None,
) -> object:
    """
    Plot a frequency-wavenumber (FK) spectrum.

    This function is intended for FK DataArrays produced by ``daspal.signal.fk``.
    It can generate either a Matplotlib plot or an interactive Holoviews
    plot.

    Parameters
    ----------
    xarr : xr.DataArray
        Input FK DataArray with ``frequency`` and ``wavenumber`` dimensions.

    ax : matplotlib.axes.Axes, default=None
        Matplotlib axis where the figure is drawn. If ``None``, a new axis
        is created.

    cmin : float, default=None
        Minimum value for the color scale.

    cmax : float, default=None
        Maximum value for the color scale.

    cbar : bool, default=True
        If True, display the colorbar.

    interactive : bool, default=False
        If True, return an interactive Holoviews plot.

    width : float, default=4
        Figure width in inches.

    height : float, default=4
        Figure height in inches.

    log_axes : bool, default=False
        If True, aA symmetric logarithmic scale (symlog) is used
        for frequency and wavenumber axes. This allows visualization of positive and
        negative ticks by keeping a linear region around zero.

    log_xmin : float, default=None
        Linear threshold around zero for the frequency axis when using a
        symmetric logarithmic scale.

    log_ymin : float, default=None
        Linear threshold around zero for the wavenumber axis when using a
        symmetric logarithmic scale.

    Returns
    -------
    object
        Matplotlib axis if ``interactive=False``.
        Holoviews object if ``interactive=True``.
    """

    fk_da = xarr.copy()

    # --- preprocessing ---
    eps = np.finfo(float).tiny   # very small float ~2e-308, to avoid log(0)
    fk = 10 * np.log10(np.maximum(np.abs(fk_da), eps))

    cmax = cmax if cmax is not None else np.amax(fk.data)
    cmin = cmin if cmin is not None else cmax - 30

    # --- holoviews interactive plot ---
    if interactive:
        if log_axes==True:
            warnings.warn(
                "Interactive plotting not available with log axes",
                UserWarning
            )
        dpi=100 # default 100 dpi value from matplotlib for comp
        width_px  = int(width * dpi)
        height_px = int(height * dpi)
        
        image = hv.Image(fk, kdims=['wavenumber','frequency'])#, label='DAS Wavenumber vs Frequency')
        return hd.rasterize(image).opts(
            clim=(cmin, cmax),
            cmap='magma_r',
            colorbar=cbar,
            #title='DAS Wavenumber vs Frequency',
            tools=['hover'],
            xlabel='Wavenumber (m⁻¹)',
            ylabel='Frequency (Hz)',
            width=width_px,
            height=height_px,
            clabel=f'dB rel. 1 {fk_da.attrs.get("unit", "")}',
            toolbar='above',
            colorbar_opts={'height':height_px//2, 'width':width_px//20}
        )
    
    # --- matplotlib plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    
    # use flipud as imshow starts in top left corner with negative frequencies
    freq = fk.frequency.values
    k = fk.wavenumber.values
    extent = [k[0], k[-1], freq[0], freq[-1]]
    img = ax.imshow(np.flipud(fk.data), vmin=cmin, vmax=cmax, cmap='magma_r',
           extent=extent, aspect='auto', interpolation='none')

    cb = None
    if cbar:
        cbar = plt.colorbar(img, ax=ax, extend="both", shrink=.5, pad=.02, aspect=15)
        cbar.set_label(f'dB rel. 1 {fk_da.attrs.get("unit", "")}')

    ax.set_xlabel('Wavenumber (m$^{-1}$)')
    ax.set_ylabel('Frequency (Hz)')

    #### option in development
    if log_axes:
        # Ploting log scale for both +/- values using symlog
        ax.set_xscale("symlog", linthresh = pow(10,log_xmin),
                      linscale=0.01, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_yscale("symlog", linthresh = pow(10,log_ymin),
                      linscale=0.01, subs=[2, 3, 4, 5, 6, 7, 8, 9])

        # defining ticks
        ximin=log_xmin+1
        yimin=log_ymin+1
        
        ximax=int(np.log10(k[-1]).round())
        yimax=int(np.log10(freq[-1]).round())
        if ximax < np.log10(k[-1]):
            ximax=ximax+1
        if yimax < np.log10(freq[-1]):
            yimax=yimax+1

        scalx=np.logspace(ximin, ximax, num=ximax-ximin+1)
        xi=np.concatenate((np.append(-1*np.flipud(scalx),0), scalx), axis=None)
        scaly=np.logspace(yimin, yimax, num=yimax-yimin+1)
        yi=np.concatenate((np.append(-1*np.flipud(scaly),0), scaly), axis=None)
        ax.set_xticks(xi)
        ax.set_yticks(yi)

        ax.set_xlim(k[0], k[-1])
        ax.set_ylim(freq[0], freq[-1])

    # scale fonts to subplot panel size
    _apply_fonts(ax, cbar=cb)
    
    return ax

####################################################################

def fft(
    xarr: xr.DataArray,
    ax=None,
    cmin: float | None = None,
    cmax: float | None = None,
    cbar: bool = True,
    log_fftaxis: bool = False,
    log_ymin: float = -3,
    d_unit: str = "m",
    interactive: bool = False,
    width: float = 4,
    height: float = 4,
) -> object:

    """
    Plot the Fourier spectrum of a DAS DataArray.

    This function is intended for FFT DataArrays produced by ``daspal.signal.fft``.
    Depending on the transformed dimension, it displays either a
    frequency-distance or time-wavenumber representation.

    Parameters
    ----------
        xarr : xr.DataArray
            Input FFT DataArray with either ``frequency`` and ``distance``
            dimensions or ``time`` and ``wavenumber`` dimensions.

        ax : matplotlib.axes.Axes, default=None
            Matplotlib axis where the figure is drawn. If ``None``, a new axis
            is created.

        cmin : float, default=None
            Minimum value for the color scale.

        cmax : float, default=None
            Maximum value for the color scale.

        cbar : bool, default=True
            If True, display the colorbar.

        log_fftaxis : bool, default=False
            If True, use a logarithmic scale for the FFT axis
            (``frequency`` or ``wavenumber``).

        log_ymin : float, default=-3
            Linear threshold used for the logarithmic scale when plotting
            wavenumber along the y-axis.

        d_unit : {"m", "km"}, default="m"
            Unit used for the distance axis when present.

        interactive : bool, default=False
            If True, return an interactive Holoviews plot.

        width : float, default=4
            Figure width in inches.

        height : float, default=4
            Figure height in inches.

    Returns
    -------
        object
            Matplotlib axis if ``interactive=False``.
            Holoviews object if ``interactive=True``.
    """
    
    fft_da = xarr.copy()

    # --- preprocessing ---
    eps = np.finfo(float).tiny   # very small float ~2e-308, to avoid log(0)
    fft = 10 * np.log10(np.maximum(np.abs(fft_da), eps))

    cmax = cmax if cmax is not None else np.amax(fft.data)
    cmin = cmin if cmin is not None else cmax - 30

    # set meta for plot type
    coords = set(fft.coords)
    if {'frequency','distance'} <= coords:
        mode = "fd"
        xname, yname = 'frequency','distance'
        xlabel, ylabel = "Frequency (Hz)", f"Distance ({d_unit})"
        title = "DAS frequency vs Distance"

        if d_unit == "km":
            fft = fft.assign_coords(distance=fft.distance/1000)
        fft = fft.sel(frequency=slice(0, None)) # keep only positive frequencies
        fft = fft.isel(frequency=slice(1, None)) # remove zero frequency
        invert_yaxis=True # holoviews param

    elif {'time','wavenumber'} <= coords:
        mode = "tk"
        xname, yname = 'time','wavenumber'
        xlabel, ylabel = "Time", "Wavenumber (m⁻¹)"
        title = "DAS Time vs Wavenumber"
        invert_yaxis=False # holoviews param
        

    else:
        raise ValueError("DataArray must contain (frequency,distance) or (time,wavenumber)")

    # --- holoviews interactive plot ---
    if interactive:
        dpi=100 # default 100 dpi value from matplotlib for comp
        width_px  = int(width * dpi)
        height_px = int(height * dpi)
        if log_fftaxis==True and mode == "fd":
            fft = fft.assign_coords(frequency=np.log10(fft.frequency))
        if log_fftaxis==True and mode == "tk":
            warnings.warn(
                "Interactive plotting not available with log of wavenumber",
                UserWarning
            )

        # Use QuadMesh for irregular grids
        image = hv.QuadMesh(fft, kdims=[xname, yname])#, label=title)
        return hd.rasterize(image).opts(
            clim=(cmin, cmax),
            cmap='viridis',
            colorbar=cbar,
            tools=['hover'],
            xlabel=xlabel,
            ylabel=ylabel,
            width=width_px,
            height=height_px,
            invert_yaxis=invert_yaxis,
            clabel=f'dB rel. 1 {fft_da.attrs.get("unit", "")}',
            toolbar='above',
            colorbar_opts={'height':height_px//2, 'width':width_px//20}
        )

    # --- matplotlib plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width,height))

    fft = fft.transpose(yname, xname)

    # "tk" mode plot
    if mode == "fd":
        frequency = fft.frequency.values
        dist = fft.distance.values
        extent = [frequency[0], frequency[-1], dist[-1], dist[0]]
        img = ax.imshow(fft.data, vmin=cmin, vmax=cmax, cmap='viridis',
                        extent=extent, aspect='auto', interpolation='none')
        ax.set_xlim(0.01 if frequency[0] < 0.01 else frequency[0], frequency[-1])

    # "tk" mode plot
    if mode == "tk":
        k = fft.wavenumber.values
        start = pd.to_datetime(fft.coords['time'][0].values)
        time_window = (fft.coords['time'][-1].values - fft.coords['time'][0].values) / np.timedelta64(1, 's')
        
        t0 = 0 if time_window <= 120 else start
        deltat = time_window if time_window <= 120 else pd.to_timedelta(time_window, unit='s')
        
        extent = [t0, t0 + deltat, k[-1], k[0]]
        img = ax.imshow(fft.data, vmin=cmin, vmax=cmax, cmap='viridis',
                        extent=extent, aspect='auto', interpolation='none')
    
        # x-axis formatting for long times
        if time_window > 120:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.set_xlabel(f"Time (s) ref. {start.strftime('%d/%m/%Y %H:%M:%S')} UTC")

    if log_fftaxis:
        if mode == "fd":
            ax.set_xscale('log')
        else:
            ax.set_yscale("symlog", linthresh = pow(10,log_ymin),
                          linscale=0.01, subs=[2, 3, 4, 5, 6, 7, 8, 9])
            # defining ticks
            yimin=log_ymin+1
            yimax=int(np.log10(k[-1]).round())
            if yimax < np.log10(k[-1]):
                yimax=yimax+1

            scaly=np.logspace(yimin, yimax, num=yimax-yimin+1)
            yi=np.concatenate((np.append(-1*np.flipud(scaly),0), scaly), axis=None)
            ax.set_yticks(yi)

            ax.set_ylim(k[0], k[-1])
        
    #ax.set_title(title)
    if xlabel != 'Time':
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cb = None
    if cbar:
        cb = plt.colorbar(img, ax=ax, extend="both", shrink=0.5, pad=0.025, aspect=15)
        cb.set_label(f'dB rel. 1 ({fft_da.attrs.get("unit", "")})')

    # scale fonts to subplot panel size
    _apply_fonts(ax, cbar=cb)

    return ax

