import os
from pathlib import Path
import numpy as np
import dask.array as da
import pandas as pd
import h5py

import daspal.core as idas
import daspal.instr.base as base
import daspal.io.h5fc as h5fc

from obspy import Trace, Stream, UTCDateTime

#############################################################################

def h5(group, path, duration, origin, date_folder, datatype, instrument: base.Instr, shots: bool, **auxmeta):
    """
    Save a single time bin of a DataArray to hdf5 file,
    Structure optionally formatted with an instrument.

    Parameters
    ----------
    group : xr.DataArray
        Subset corresponding to a time bin.
    path : str or Path
        Output folder path.
    duration : int
        Bin duration in seconds.
    origin : bool
        create file at exact time or rounded value
    date_folder: str
        date subfolder
    datatype : str
        Subfolder or type for saving.
    instrument : instrument object and use .format().
        If not provided or .format() does not exist, falls back to group.attrs.
    shots : if True add the shot number in filename
    **auxmeta : directly pass new metadata to be integrated within the saved file
        e.g. fibre=F7
    """
    
    if group.time.size == 0:
        return

    dt_sample = group.attrs['dt']
    # Lazy-safe min/max
    time0 = pd.to_datetime(group.time.min().item())
    timeN = pd.to_datetime(group.time.max().item())

    actual_duration = float((timeN - time0) / np.timedelta64(1, 's'))
    if actual_duration < duration - dt_sample:
        return
    
    group.attrs = idas.adjust_metadata(group)
    group.attrs['time'] = time0.value/1e9

    # Build file path
    if date_folder:
        filepath = Path(path) / time0.strftime("%Y%m%d") / datatype / f"{time0.strftime('%H%M%S')}.hdf5"
    elif origin == "exact":
        prefix = f"{group.attrs['shot_number']}_" if shots else ""
        filepath = Path(path) / f"{prefix}{time0.strftime('%Y%m%dT%H%M%S')}.{time0.microsecond:06d}Z.hdf5" # microseconds precision
   
    # Make sure parent directories exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Format metadata to instrument
    ffiddict = instrument.format(group, **auxmeta)

    # Keep data lazy
    ffiddict['data'] = group.data

    
    # Save
    with h5py.File(filepath, 'w') as f:
        h5fc.save_h5fields(f, ffiddict)


#############################################################################

def mseed(dataset, out_path, net='XX', cha='HSF'):
    """
    Convert DAS data blocks into MiniSEED files by distance (station).

    This function handles both:
      - A single Dask-backed xarray.DataArray
      - A list of Dask-backed xarray.DataArray blocks (continuous blocks separated by gaps)

    Each distance coordinate becomes a separate Trace in a Stream. 
    Blocks are added sequentially to preserve data gaps.

    Parameters
    ----------
    dataset : xarray.DataArray or list of xarray.DataArray
        Input DAS data block(s). Each block must have coordinates 'time' and 'distance',
        and attributes 'time' (start timestamp) and 'dt' (sampling interval).
    out_path : str
        Root output path for MiniSEED files.
    net : str, optional
        Network code for MiniSEED (default 'XX').
    cha : str, optional
        Channel code for MiniSEED (default 'HSF').

    """

    # Ensure dataset is always a list of data array
    if not isinstance(dataset, list):
        dataset = [dataset]

    start_time = UTCDateTime(dataset[0].attrs['time'])
    year = start_time.year
    julday = start_time.julday

    # Info: report number of blocks and stations
    n_blocks = len(dataset)
    distances = dataset[0].coords['distance'].data
    n_distances = len(distances)
    print(f"Converting {n_blocks} time block(s) for {n_distances} distance(s) to MiniSEED stations.")

    for d in distances:
        stream = Stream()
        ch_id = int(d/dataset[0].attrs['dxch'])

        out_dir = f"{out_path}/{year}/{net}/{ch_id:05d}/{cha}.D"
        os.makedirs(out_dir, exist_ok=True) # Make root directory

        for block in dataset:
            xarr = block.sel(distance=d)
            start_time = UTCDateTime(xarr.attrs['time'])
            stats = {"network": net,
                "station": f"{ch_id:05d}",
                "location": '',
                "channel": cha,
                "starttime": start_time,
                "sampling_rate": 1 / xarr.attrs['dt']}

            data = xarr.data
            if isinstance(data, da.Array):
                data=data.compute()
            trace = Trace(data=data, header=stats)                
            stream += trace   
        
        # save data
        stream.write(f"{out_dir}/{net}.{ch_id:05d}..{cha}.D.{year}.{julday}.mseed", format="MSEED")