"""
Library for setting up xarray accessors on the main modules
`plot`, `process`, `signal`
"""

import functools
import xarray as xr
import daspal.plot as plot
import daspal.process as process
import daspal.signal as signal

def _wrap(func):
    """Return a bound-method wrapper that prepends self._obj as first argument."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self._obj, *args, **kwargs)
    return wrapper

@xr.register_dataarray_accessor("daspal")
class DaspalAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # ------------------------------------------------------------------ #
    #  signal                                                              #
    # ------------------------------------------------------------------ #
    taper             = _wrap(signal.taper)
    demean            = _wrap(signal.demean)
    detrend           = _wrap(signal.detrend)
    hilbert           = _wrap(signal.hilbert)
    filter            = _wrap(signal.filter)
    fft               = _wrap(signal.fft)
    ifft              = _wrap(signal.ifft)
    fk                = _wrap(signal.fk)
    ifk               = _wrap(signal.ifk)
    fk_filt           = _wrap(signal.fk_filt)
    rms               = _wrap(signal.rms)
    psd               = _wrap(signal.psd)
    spectrograms      = _wrap(signal.spectrograms)

    # ------------------------------------------------------------------ #
    #  process                                                             #
    # ------------------------------------------------------------------ #
    resample          = _wrap(process.resample)
    select_channels   = _wrap(process.select_channels)
    cmr               = _wrap(process.cmr)
    DASinstr          = _wrap(process.DASinstr)

    # ------------------------------------------------------------------ #
    #  view  (prefixed with plot_ to avoid collision with signal names)   #
    # ------------------------------------------------------------------ #
    plot_das          = _wrap(plot.das)
    plot_psd          = _wrap(plot.psd)
    plot_spectrogram  = _wrap(plot.spectrogram)
    plot_fk           = _wrap(plot.fk)
    plot_fft          = _wrap(plot.fft)
