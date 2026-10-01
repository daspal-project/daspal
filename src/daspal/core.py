"""
Core library of `daspal` package for selecting, reading and saving DAS data
"""


from pathlib import Path
from collections import Counter

from dask import delayed, compute, persist
import dask.array as da
from dask.distributed import Client, get_client

import datetime
import numpy as np
import pandas as pd
import xarray as xr

import daspal.io.finder as finder
import daspal.io.read as read
import daspal.io.write as write
import daspal.process as process
import daspal.dask_utils.tasks as tasks
import daspal.instr.registry as registry
from daspal.io import zarrfc

from daspal._logger import logger

#######################################################################################
################################ Finding data #########################################
#######################################################################################

def find_files(
    path: str | Path,
    start: datetime.datetime,
    end: datetime.datetime,
    instr: str | None = None,
    datatype: str = "dphi",
    file_length: int | None = None,
    check_gaps: bool = True,
) -> list[str] | dict:

    """
    Find and organize DAS data files within a time range.

    This function uses the ``Finder`` class from ``daspal.io.finder`` to scan
    a directory hierarchy and identify files matching the requested instrument,
    datatype, and time window.

    It can detect temporal gaps and return either a continuous file list or
    separate blocks of contiguous files.

    Parameters
    ----------
        path : str or pathlib.Path
            Base directory containing the DAS data files.

        start : datetime.datetime
            Start time of the requested data window (inclusive).

        end : datetime.datetime
            End time of the requested data window (exclusive).

        instr : str, default=None
            Instrument identifier used to select the appropriate file format.

        datatype : str, default="dphi"
            Data type to search for (e.g. ``"dphi"`` or ``"proc"``).

        file_length : int, default=None
            Expected duration of each file in seconds. If ``None``, the default
            file length for the instrument is used.

        check_gaps : bool, default=True
            If True, detect gaps in the time coverage and split files into
            contiguous blocks.

    Returns
    -------
        list[str] or dict
            File paths matching the requested time window.

            If no gaps are detected, returns a continuous list of file paths.

            If gaps are present and ``check_gaps=True``, returns a dictionary
            containing separate contiguous file blocks.

    Notes
    -----
        This function does not load the data; it only finds and organizes file
        paths.

        Gap detection is recommended before loading data to correctly handle
        discontinuous time series.
    """

    try:
        instrument = registry.get_instrument(instr)
    except KeyError:
        raise ValueError(
            f"`instr` needs to be defined as: {', '.join(INSTRUMENTS.keys())}\n"
            f"if data formatted to daspal format use 'instr=dasproc'"
        )

    f = finder.Finder(path, start, end, instrument, datatype, file_length, check_gaps)

    return f.find_files()


#######################################################################################
################################ Reading data #########################################
#######################################################################################

def _load_single_block(
    paths: list[str] | str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    instr: str | None = None,
    dist_sel: tuple | list | None = None,
    meta: dict[str, str] | None = None,
    opt_read: bool = True,
    lazy_read_boost: bool = True
) -> xr.DataArray | None:
    """
    Load a single contiguous block of DAS data files into a Dask-backed xarray.DataArray.

    Parameters
    ----------
        paths : str or list of str
            File path(s) for a contiguous block.
        start, end : datetime, optional
            Time slice to apply after loading.
        instr : str
            Instrument type.
        dist_sel : list or tuple of length 2 for selecting distance limits
            optional, use [dmin, dmax) or None for no limit.
        meta : dict
            option to pass on metadata at data loading (e.g. 'project', 'cable')
        opt_read : bool, default=True
            If True, group/split files into batches optimized for the target
            chunk size before reading (see ``daspal.dask_utils.tasks.opt_read``).
            If False, each file is read as its own batch (no size-based
            regrouping), which is useful e.g. to study the effect of the
            natural file-based chunking on memory usage.
        lazy_read_boost : bool, default=True
            If True, assumes all files have the same time length as the first file,
            avoiding per-file metadata reads.
            If False, the time dimension of each file is read from its metadata when
            the Dask task graph is built. This provides a check for files with
            different time lengths, but can slow down graph construction when a very
            large number of files is requested (e.g. when creating long spectrograms).

    Returns
    -------
        xr.DataArray
            Dask-backed DataArray for one contiguous block.

    """

    if paths==[]:
        return
    
    # handle single file read
    paths = [paths] if isinstance(paths, str) else paths
    
    # assign instrument
    try:
        instrument = registry.get_instrument(instr)
    except KeyError:
        raise ValueError(
            f"`instr` needs to be defined as: {', '.join(INSTRUMENTS.keys())}\n"
            f"if data formatted to daspal format use 'instr=dasproc'"
        )

    ##### optimization #####
    # extract metadata info from first file read and check data size to be read
    read_info, file_size, size_read_per_file = read.file_info(paths[0], dist_sel, instrument)
    # add aditionnal info to be passed druing data reading process
    read_info["dist_sel"] = dist_sel
    read_info["instrument"] = instrument # base.Instr
    
    read_info["lazy_read_boost"] = lazy_read_boost # dask graph optimization

    if opt_read:
        # split or not the file list for optimized read with dask
        batch_paths = tasks.opt_read(paths, file_size, size_read_per_file)
    else:
        # no size-based batching: one batch per file
        batch_paths = [[p] for p in paths]
        
    # read the data
    try:
        client = get_client()
        if hasattr(client, "cluster") and client.cluster is not None:
            workers_info = client.cluster.worker_spec
            nb_workers = len(workers_info)
        else:
            nb_workers = 1  # fallback, real client/workers determined at compute
    except ValueError:
        client = None  # no active client, will use default at compute
        nb_workers = 1  # fallback, real client/workers determined at compute
   
    if isinstance(client, Client) and client.status == "running" and len(batch_paths) > 2*nb_workers:
        # submit batches to workers for faster graph creation
        # but only if a certain number of batches
        futures = [client.submit(read.files_batch, batch, read_info) for batch in batch_paths]
        objs = client.gather(futures)
    else:
    # no client is provided, lazy objects created sequentially
    # in the main Python thread. When data.compute() is called,
    # Dask use its default "threads" scheduler to run
    # tasks in parallel on local CPU cores
        objs = [read.files_batch(batch, read_info) for batch in batch_paths]

    # concatenate everything lazily
    data = xr.concat(objs, dim="time")
    if lazy_read_boost:
        time_axis = read_info["t0"] + pd.timedelta_range(
            start="0s",
            periods=data.sizes["time"],
            freq="%dns" % int(read_info["dt"] / np.timedelta64(1, "ns") + 0.5),
        )
        data = data.assign_coords(time=time_axis)
    else:
        data = data.sortby("time")
    
    # precise time cut
        # up to here files are read fully so data not cut precisely for start and end
    if start is not None and end is not None:
        data = data.sel(time=(data.time >= np.datetime64(start))& (data.time < np.datetime64(end)))
    else:
        logger.warning("no `start` and `end` time input specified in `load_data`, all files in `paths` are loaded fully")

    # safety net if after the time cut, data becomes empty
    if data.size == 0:
        logger.warning("DataArray is empty")
        return
    
    ############ metadata ################
    # add metadata to datarray
    data.attrs = instrument.metadata(paths[0], meta)
    # correct metadata based on data read size
    data.attrs = adjust_metadata(data)

    # format metadata to `dasproc` standard
    data.attrs = registry.get_instrument("dasproc").format(data)

    # make metadata zarr compatible
    data.attrs = zarrfc.meta_comp(data.attrs)
    
    return data

#####################################

def load_data(
    paths: list[str] | dict | str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    instr: str | None = None,
    dist_sel: np.ndarray | None = None,
    preproc: bool = True,
    meta: dict[str, str] | None = None,
    opt_read: bool = True,
    lazy_read_boost: bool = True,
    **kwargs

) -> xr.DataArray | list[xr.DataArray]:

    """
    This function reads HDF5 files to load DAS data,
    optionally selecting a time window and distance range.

    Assumes `paths` are already validated and sorted by Finder.

    Wraps `_load_single_block` and supports data with temporal gaps. 
    If `paths` is a single contiguous list of files, a single DataArray is returned, 
    but if gaps exist (i.e., `paths` is a dict of blocks), a list of DataArrays is returned,
    each corresponding to a contiguous time block.
    
    Each block is Dask-backed; computation is lazy.
    When multiple blocks are returned, functions must be applied per block or after merging.

    Parameters
    ----------
        paths : str, list of str, or dict
            File path(s) or blocks of file paths. If dict, each key represents a continuous block.
        start : datetime.datetime, optional
            Lower time limit for data selection (inclusive). Default is None.
        end : datetime.datetime, optional
            Upper time limit for data selection (exclusive). Default is None.
        instr : str, optional
            Instrument type. Supported: `'optodas'`, `'dxs'`, `'apsensing'`.
        dist_sel : np.ndarray, optional
            Distance range `[dmin, dmax]` to load (dmax excluded).
        preproc : bool, optional (so far only for optodas)
            Pre-process data based on instrument specs. Default is True.
            See **kwargs for options.
        meta : dict
            option to pass on metadata at data loading (e.g. 'project', 'cable')
        opt_read : bool, default=True
            If True, batch files by target read size for optimized parallel
            reading. If False, each file is read as its own batch, useful for
            benchmarking the impact of file-based chunking on memory usage.
        lazy_read_boost : bool, default=False
            If True, assumes all files have the same time length as the first file,
            avoiding per-file metadata reads. If False, the time dimension of each file
            is read from its metadata when the Dask task graph is built.
        
        **kwargs
            Optional keyword arguments passed to ``process.DASinstr`` when
            ``preproc=True``:

            ``strain`` (bool, default: ``True``)
                Convert to strain.

            ``unwrap`` (bool, default: ``False``)
                Unwrap phase.

            ``integrate`` (bool, default: ``False``)
                Integrate the signal.

    Returns
    -------
        xr.DataArray or list[xr.DataArray]
            A single DataArray for contiguous data, or a list of DataArrays
            when temporal gaps are present.

    Examples
    --------
        Single contiguous block

        >>> import daspal.signal as signal
        >>> data = load_data(filepaths, start=START, end=END)
        >>> processed_data = signal.filter(data, fcut=(2, 10), filt_type='bandpass')

        Data with gaps

        >>> import daspal.process as process
        >>> import daspal.signal as signal
        >>> dataset = load_data(filepaths_with_gaps)  # returns list of DataArray
        >>> processed_dataset = process.blocks(dataset, signal.filter, fcut=(2, 10), filt_type='bandpass')
    """

    if isinstance(paths, dict):
        dataset = []
        for block_name, block_paths in paths.items():
            #print(block_name, block_paths)
            data = _load_single_block(
                block_paths, start=start, end=end, 
                instr=instr, dist_sel=dist_sel, meta=meta,
                opt_read=opt_read, lazy_read_boost=lazy_read_boost)
            if data is not None:
                dataset.append(data)
            
        if not dataset:
            return None
        
        logger.warning(
            f"Data is a list of {len(dataset)} blocks due to gaps. "
            f"Cannot apply accessors on `dataset` directly. "
            f"Functions should be applied per block using process.blocks or use core.merge function."
        )
        ####### preprocessing ########
        if preproc:
            dataset = process.blocks(dataset, process.DASinstr,
                                     instr=instr, **kwargs)
        return dataset

    else:
        data = _load_single_block(
            paths, start=start, end=end,
            instr=instr, dist_sel=dist_sel, meta=meta,
            opt_read=opt_read, lazy_read_boost=lazy_read_boost)
        ####### preprocessing ########
        if preproc and data is not None:
            data = process.DASinstr(data, instr=instr, **kwargs)
        return data
        
#####################################

def merge(dataset: list[xr.DataArray], chunk_size=20_000) -> xr.DataArray:

    """
    Merge a dataset composed of multiple contiguous data blocks into a
    single time-continuous DataArray.

    Temporal gaps between blocks are filled with NaN values to preserve
    continuity along the time axis.

    Parameters
    ----------
        dataset : list[xr.DataArray]
            List of DataArrays representing contiguous blocks of data.
        chunk_size : int, optional
            Number of time samples per chunk in the output DataArray.
            Defaults to 20,000.

    Returns
    -------
        xarray.DataArray
            A single time-continuous DataArray with NaN-filled gaps between
            blocks.
    """

    def time_chunks(t0, t1, step, chunk_size):
        # TIME CHUNK GENERATOR
        while t0 < t1:
            t_end = min(t1+step, t0 + step * chunk_size)
            yield np.arange(t0, t_end, step, dtype="datetime64[ns]")
            t0 = t_end
    
    # Combining dataset and assigning NaN values for gaps
    def build_chunk(t_chunk):
        # slice through dataset for t_chunk time window
        data_subsets = [ds.sel(time=slice(t_chunk[0], t_chunk[-1])) for ds in dataset]
        overlap_subsets = [ds for ds in data_subsets if ds.time.size > 0] # non-empty slices

        if overlap_subsets:
            merged = xr.concat(overlap_subsets, dim='time').reindex(time=t_chunk)
        else:
            template = dataset[0].isel(time=0, drop=True)
            # build lazy nan chunk
            nan_lazy = da.full_like(template.data, fill_value=np.nan, dtype=np.float32)
            nan_lazy_expanded = da.broadcast_to(nan_lazy, (len(t_chunk),) + template.shape)
            merged = xr.DataArray(nan_lazy_expanded, dims=('time',) + template.dims, 
                                  coords={'time': t_chunk, **template.coords})
        return merged

    # Determine the most common time chunk size
    chunks = []
    for ds in dataset:
        chunks.extend(ds.chunksizes["time"])
    chunk_size = Counter(chunks).most_common(1)[0][0]

    # Build continuous time array
    t0 = min(ds.time.values.min() for ds in dataset)
    t1 = max(ds.time.values.max() for ds in dataset)
    dt_ns = np.timedelta64(int(dataset[0].attrs['dt'] * 1e9), "ns")

    #Use time_chunk generator to avoid building the whole array at once
    data = xr.concat((build_chunk(t_chunk) for t_chunk in time_chunks(t0, t1, dt_ns, chunk_size)), dim="time")

    data = data.chunk({"time": chunk_size})
    
    return data

def partition(
    dataset: xr.DataArray | list[xr.DataArray],
    duration: int = 60,
    origin: str = "exact",
) -> list[xr.DataArray]:
    """
    Split a DataArray or dataset into fixed-duration time blocks.

    Parameters
    ----------
        dataset : xr.DataArray or list[xr.DataArray]
            DataArray or list of DataArrays (including Dask-backed arrays)
            containing a ``time`` dimension and a ``dt`` attribute.

        duration : int, default=60
            Duration of each output block, in seconds.

        origin : {"exact", "rounded"}, default="exact"
            Determines how the start time of the partitioning is aligned.

            - ``"exact"``: Preserve the original start time.
            - ``"rounded"``: Round the start time up to the next multiple of
            ``duration``. For example, ``09:02:10`` becomes ``09:05:00``
            when ``duration=300``.

    Returns
    -------
        list[xr.DataArray]
            List of DataArrays with a duration of ``duration`` seconds.
    """

    # Ensure dataset is always a list of data array
    if not isinstance(dataset, list):
        dataset = [dataset]

    groups = []
    
    for xarr in dataset:  # dataset is a list of DataArrays

        dt_sample = xarr.attrs['dt']
        n_samples_per_bin = int(duration / dt_sample)
        
        # --- align start/end times ---
        start_time = pd.to_datetime(xarr.time.min().item())
        end_time   = pd.to_datetime(xarr.time.max().item())

        # Align start time per block
        if origin == "rounded":
            aligned_start = start_time.ceil(pd.Timedelta(seconds=duration))
        elif origin == "exact":
            aligned_start = start_time
        else:
            raise ValueError(f"origin must be 'exact' or 'rounded'")

        # skip blocks shorter than one usable bin
        if (end_time - aligned_start) < (pd.Timedelta(seconds=duration) - pd.Timedelta(seconds=dt_sample)):
            #print("Block shorter than one usable bin → skipping block")
            continue

        # --- create time bins ---
        bin_starts = pd.date_range(
            start=aligned_start,
            end=end_time,
            freq=pd.Timedelta(seconds=duration)
        )

        for t0 in bin_starts:
            t1 = t0 + pd.Timedelta(seconds=duration)
            #group = xarr.sel(time=slice(t0, t1))
            # below approach more strict on excluding t1
            group = xarr.sel(time=(xarr.time >= np.datetime64(t0))& (xarr.time < np.datetime64(t1)))
            if group.sizes['time'] == n_samples_per_bin:  # only keep full bins
                groups.append(group)
    return groups

    
#############################################################################

def adjust_metadata(dataarray: xr.DataArray) -> dict:
    """
    adjust metadata on newly generated xr.DataArray
    """
    m = dataarray.attrs

    # update dx to distance between measured channels (optodas)
    m['dx'] = dataarray.coords['distance'][1].values-dataarray.coords['distance'][0].values
    m['time'] = np.float32(dataarray.time[0].values.astype('datetime64[ns]').astype('int64')) / 1e9
    m['distance']=np.float32(dataarray.distance[0])
    
    return m

def flatten_meta(m) -> dict:
    """
    takes `dasproc` metadata format as input and flattens it out
    """
    
    m_out = {}

    for k, v in m.items():
        if k == "instrument":
            m_out["instrument"] = v["type"]
            m_out.update(v.get("parameters", {}))
        else:
            m_out[k] = v

    return m_out

#######################################################################################
################################# Saving data #########################################
#######################################################################################

def save(
    dataset: xr.DataArray | list[xr.DataArray],
    path: str,
    duration: int,
    origin: str = "exact",
    date_folder: bool = True,
    datatype: str = "proc",
    instr: str = "dasproc",
    shots: bool = False,
    **auxmeta,
):
    """
    Save a dataset into HDF5 files split into time bins.

    The dataset can be a single DataArray or a list of DataArrays. Output
    files are created according to the specified ``duration`` and
    ``origin`` parameters.

    Parameters
    ----------
        dataset : xr.DataArray or list[xr.DataArray]
            Input DAS data block(s). Each DataArray must contain ``time`` and
            ``distance`` coordinates, and ``time`` and ``dt`` attributes.

        path : str
            Output folder path.

        duration : int
            Duration of each saved time bin, in seconds.

        origin : {"exact", "rounded"}, default="exact"
            Alignment mode for the start time of each time bin.

            - ``"exact"``: Preserve the original start time.
            - ``"rounded"``: Round the start time up to the next multiple of
            ``duration``. For example, ``09:02:10`` becomes ``09:05:00``
            when ``duration=300``.

        date_folder : bool, default=True
            If True, organize files into date-based folders with ``datatype``
            subfolders. If False, save files directly under ``path``.

        datatype : str, default="proc"
            Data type or subfolder name used in the output structure.

        instr : str, default="dasproc"
            Output format and instrument metadata structure. By default, uses
            the native ``dasproc`` format.

        shots : bool, default=False
            If True, include the shot number in the file name. The ``shot number``
            label must be present in the DataArray attributes.

        **auxmeta
            Additional metadata to add to the saved file.

            Example:
                ``fibre="F7"``
    """

    try:
        instrument = registry.get_instrument(instr)
    except KeyError:
        raise ValueError(
                f"`instr` needs to be defined as: {', '.join(INSTRUMENTS.keys())}"
            )

    # Ensure the directory exists and if not create it
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- partition dataset into bins of fixed duration ---
    groups = partition(dataset, duration=duration, origin=origin)
    
    # --- delayed writing + compute bins ---
    if isinstance(dataset[0].data, da.Array):
        # ensure groups are persisted to avoid competing tasks with saving
        persisted_groups = persist(*groups)
        # delayed call to write_to_h5 + compute
        batch = [delayed(write.h5)(g, path, duration, origin, date_folder, datatype, instrument, shots, **auxmeta) for g in persisted_groups]
        compute(*batch)
    else:
        # fully loaded arrays: just call h5() directly
        for g in groups:
            write.h5(g, path, duration, origin, date_folder, datatype, instrument, shots, **auxmeta)
