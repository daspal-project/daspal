
from dask.distributed import Client, get_client
import numpy as np
import xarray as xr

from daspal._logger import logger

#######################################################################################
#################### functions for Dask optimisation ##################################
#######################################################################################

def opt_read(fnames, file_size, size_read_per_file):
    """
    Optimize tasks associated with files read.
    Split a list of files into batches for efficient reading, optimized for Dask parallelism.

    Parameters
    ----------
    fnames : list[str]
        Path to the data files, determined after using function `find_files`.
    file_size : float
        Size of one file on disk (MB). Used to determine I/O strategy.
    size_read_per_file : float
        Size of data actually read into memory per file after slicing (MB).

    Returns
    -------
    list[list[str]]
        List of file batches, each batch being a list of file paths.
    
    """
    
    nb_files = len(fnames)
    total_mem_read = nb_files * size_read_per_file
    
    try:
        client = get_client()
    except ValueError:
        client = None

    if client is None or client.status != "running":
        files_per_batch = 1
        return [[f] for f in fnames]  # one file per batch as safe fallback

    # query workers / total RAM
    workers_info = client.scheduler_info()["workers"]
    nb_workers = len(workers_info)
    total_ram = sum(w["memory_limit"] for w in workers_info.values()) / 1024**2  # MB

    if total_mem_read > 0.7 * total_ram:
        logger.warning(
            f"If no data resampling performed the request will exceed memory limits.\n"
            f"Total requested={total_mem_read:.1f} MB  > allowed={0.7 * total_ram:.1f} MB."
        )
    
    # Places files into batches under certain conditions
    FILE_SIZE_THRESHOLD_MB = 200  # threshold on file size
    BATCH_MEM_LIM_MB = 150        # max data per task (defines chunk size)

    if file_size > FILE_SIZE_THRESHOLD_MB:
        # large files: limit I/O per task
        files_per_batch = 1
    else:
        # smaller files: how many files needed to reach target batch size
        files_per_batch_mem = max(1, int(BATCH_MEM_LIM_MB // size_read_per_file))
        files_per_batch = min(files_per_batch_mem, 10) # cap serial I/O

    files_batches = [fnames[i:i + files_per_batch] for i in range(0, len(fnames), files_per_batch)]

    logger.info(
        f"Files handling:\n"
        f"total files: {nb_files}, "
        f"files per batch: {files_per_batch}, "
        f"approx read size per batch: {files_per_batch * size_read_per_file:.1f} MB",
    )
    
    return files_batches

#####################################

def opt_process(results, batch_size, max_tasks = 50000):
    """
    Dynamically adjusts batch size (e.g. number of time windows to be processed)
    based on an estimation of Dask task graph size.

    This function estimates the number of tasks associated with a batch by combining
    the Dask task graphs of all provided results, then scales the batch size
    so that the total number of tasks per batch does not exceed `max_tasks`.

    Parameters
    ----------
    results : list
        List of Dask-backed objects representing the current batch outputs
    batch_size : int
        Current number of time windows per batch
    max_tasks : int, optional
        Maximum allowed number of Dask tasks per batch

    Returns
    -------
    int
        Updated batch size for the next iteration.

    Note
    ----
        This function is used within the function `utils.process_times`
        in order to optimize data processing over large time windows
        
    """

    # combine all task graphs from the results and count the unique tasks
    graphs = {}

    for result in results or []: # handles results begind None
        if result is None:
            continue

        if hasattr(result, "__dask_graph__"): # check if the object supports Dask graphs
            graph = result.__dask_graph__() # get the object's task graph
            if graph is not None:
                graphs.update(graph)

    tasks = len(graphs)

    if tasks == 0:
        return batch_size
    
    #print(f"tasks={tasks}")

    # updating batch size dynamically
    tasks_per_batch = tasks / batch_size # cost per single time window
    next_batch_size = max(1, int(max_tasks / tasks_per_batch)) # batch size scaled with defined maximum tasks

    return next_batch_size


#####################################

def rechunk_along_axis(dataarray: xr.DataArray, axis=-1):
    '''
    Function to rechunk dataarray to a single chunk along a defined axis.
    This is useful for FFT or iFFT transforms that require a fully
    continuous axis.
    However, this will generate shuffle operations and it is better 
    to have chunk sizes smaller (<10MB) for shuffle operations.
    e. g. reducing chunk sizes 
    Here the ``other axis` chunk size is for now divided by 4 prior to the -1 rechunk
    but can be coptimised.

    Parameters
    ----------
        xarray.DataArray: input DataArray with dimensions 'time' and 'distance'
        axis (int): axis along which to perform the transform (default is last axis)
        
    Returns
    -------
        xarray.DataArray: DataArray with dimensions 'time' and 'distance'

    used in signal.fft and signal.ifft functions

    '''
    
    dims = dataarray.dims
    axis = axis if axis >= 0 else len(dims) + axis  # convert -1 axis to last
    other_axis = 1 - axis
    other_dim = dims[other_axis]

    orig_chunk = dataarray.chunks[other_axis][0] # first chunk size
    new_chunk = orig_chunk // 4
    #if orig_chunk % 4 != 0:
    #    logger.warning(f'first chunk size {orig_chunk} for dimension {other_axis} is an odd number, chunk division by 4 to optimise shuffling may be imprecise', stacklevel=2)

    dataarray = dataarray.chunk({other_dim: new_chunk})
            
    # rechunk to a single chunk along the axis, keeping other to `auto`
    chunks = ['auto'] * dataarray.ndim
    chunks[axis] = -1
    
    return xr.DataArray(dataarray.data.rechunk(tuple(chunks)),
                        dims=dataarray.dims, coords=dataarray.coords, attrs=dataarray.attrs)
