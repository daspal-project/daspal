"""
Library including util functions for processing DAS data
"""

import numpy as np
import datetime
from daspal.core import adjust_metadata

import daspal.dask_utils.tasks as tasks

from dask import compute
from dask.distributed import get_client

from daspal._logger import logger

from typing import Iterator, Callable

#############################################################################
######################## extract time windows ###############################

def gen_time_partitions(
    start: datetime.datetime | np.datetime64,
    end: datetime.datetime | np.datetime64,
    time_window: int | float,
    overlap: float = 0
) -> Iterator[tuple[datetime.datetime | np.datetime64, datetime.datetime | np.datetime64]]:

    """
    Partition a time range into overlapping windows.

    Generates consecutive time windows between `start` and `end`, with an
    optional fractional overlap between adjacent windows. Supports both
    datetime.datetime and np.datetime64 inputs.

    Parameters
    ----------
        start : datetime or np.datetime64
            Start time of the full range.
        end : datetime or np.datetime64
            End time of the full range.
        time_window : int or float
            Duration of each window in seconds.
        overlap : float
            Fraction of `time_window` to overlap between consecutive windows.
            Must be between 0 and 1. Default is 0 (no overlap).

    Yields
    ------
        tuple of (datetime or np.datetime64, datetime or np.datetime64)
            Start and end times of each generated window.
    """

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")
    
    # Detect type
    is_np = isinstance(start, np.datetime64)
    
    # Convert time_window to timedelta
    t_win = datetime.timedelta(seconds=time_window) if not is_np else np.timedelta64(int(time_window), 's')
    step  = t_win * (1 - overlap)
    
    start_ = start
    while start_ < end:
        end_ = start_ + t_win  # help solve (END - START) not perfectly divisible by time_window
        if end_ >= end:  # last window
            end_ = end
            yield start_, end_
            break
        yield start_, end_
        start_ += step

#############################################################################

def create_day_windows(
    start: datetime.datetime,
    end: datetime.datetime
) -> Iterator[tuple[datetime.datetime, datetime.datetime]]:

    """
    Generate daily time windows within a defined time range.

    Parameters
    ----------
        start : datetime
            Start time of the full range.
        end : datetime
            End time of the full range.

    Yields
    ------
        tuple of (datetime, datetime)
            Start and end times of each daily window.
    """

    start_ = start
 
    while start_ < end:
        end_ = min(datetime.datetime(start_.year, start_.month, start_.day)
                    +datetime.timedelta(days=1), end)
        yield start_, end_
        start_ = end_


#############################################################################
######################## process time windows ###############################

def process_times(
    times: list[tuple[datetime.datetime, datetime.datetime]],
    process_func: Callable,
    batch_size: int = 1,
    max_tasks: int = 50000,
    **kwargs
) -> list:

    """
    Processes a sequence of time windows in adaptive batches using a user-defined function.

    The function splits `times` into batches of time windows, applies `process_func`
    to each batch, dynamically adjusts batch size based on computational cost
    (via Dask task graph estimation), and aggregates the results.

    Parameters
    ----------
        times : list of tuple of datetime
            List of time window intervals. Each tuple contains
            (start_time, end_time).

        process_func : callable
            Function applied to each batch of time windows. Must accept a list
            of time windows as its first argument and return Dask-lazy results.

        batch_size : int
            Initial number of time windows per batch. Dynamically updated during
            processing. Default is 1.

        max_tasks : int
            Maximum allowed number of Dask tasks per batch. Default is 50000.

        **kwargs : dict
            Additional keyword arguments passed to `process_func`.

    Returns
    -------
        list
            Aggregated results from all batches. Each inner list corresponds to one
            output stream from `process_func`, containing computed results across
            all processed batches.

    Notes
    -----
        - Batch size is dynamically adjusted using Dask task graph estimation.
        - Computation is triggered per batch.
        - Each output object is annotated with a `dt` attribute corresponding
          to the duration of one time window.
    """

    client = get_client()
    
    outputs = None
    i=0
    while i < len(times):
        batch = times[i : i + batch_size]
        print(
            f"Processing batch: "
            f"{batch[0][0]:%Y-%m-%d %H:%M:%S} -> "
            f"{batch[-1][1]:%Y-%m-%d %H:%M:%S}"
        )
        logger.info(f"batch_size={len(batch)}")

        # apply function to batch
        results = process_func(batch, **kwargs)

        # optimize windowing lazy processing
        batch_size = tasks.opt_process(results, batch_size, max_tasks = max_tasks)

        # compute all lazy values for this batch
        results = compute(*results)

        ## initiate output lists based on number of results
        if outputs is None:
            outputs = [[] for _ in results]

        # update dt attrs and dynamically store data in datasets
        for data, store in zip(results, outputs):
            data.attrs["dt"] = (times[0][1] - times[0][0]).total_seconds()
            store.append(data)
            
        i += len(batch)  # update i based on the last batch size

    return outputs

#######################################################################################

#def process_distances(...)

#############################################################################

def correct_dt(xarr, true_dt, start, end):
    """
    Function to correct the data if wrong dt written to file
    created following a bug on optodas processing tree where
    time decimation was not taken into account for the dt
    recording in the metadata
    
    to be used without sepcifying start and end in load_data
    """

    xarr.attrs['dt'] = true_dt

    t = np.linspace(0, xarr.sizes['time']*xarr.attrs['dt'],
            xarr.sizes['time'], endpoint=False)
    time = xarr.coords['time'][0].values + np.array(t * 1e9, dtype='timedelta64[ns]')
    xarr.coords['time'] = time

    m = adjust_metadata(xarr)
    xarr.attrs = m

    # recut the data
    xarr = xarr.sel(time=(xarr.time >= np.datetime64(start))& (xarr.time < np.datetime64(end)))

    return xarr

