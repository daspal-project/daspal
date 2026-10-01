# User Guide

::::{tab-set}
:::{tab-item} Data structure
For compatibility and inter-library integration, the library `Xarray` was considered to be the main backbone of the data structure. It is also well integrated with Dask but also with data storage format such as `HDF5`, `NetCDF` or `Zarr` when it comes to efficiently save and access data. The idea here is to directly work with Xarray which offer broad out-of-the-box compatibility, with `xarray.DataArray` giving access to the full xarray API. In addition, DAS-specific methods have been added on top via xarray customed `daspal` accessors, keeping a clean and consistent interface without modifying the core object and usable alongside standard xarray accessors. Metadata via attributes is also directly stored in the `DataArray` object for easy access.
:::

:::{tab-item} Working with Dask
For optimized data analysis, Dask offers great scalability. A DAS processing pipeline can be tailored for a simple laptop or, when large datasets are involved, scaled to be dispatched across a cluster of multiple machines.

Loading data with `daspal` creates a virtual dataset with data divided in chunks for efficent parallelization. Dask enables the build of a task graph that is usually labeled as a lazy execution, meaning each tasks of a specific workflow gets registered or scheduled but are not computed yet. This enables performance tuning of the different tasks that compose the task graph for executing them concurrently. Tasks can correspond to simple data access or more advance processing steps. Results are then be computed (processed and loaded in memory) at the end or at any requested intermediate steps.

One of the strength of Dask is the use of its dashboard. The dashboard gives real-time visual performance assessment of any Dask-backed workflow by highlighting for example: the memory usage per workers, the number of tasks being processed and particularly how those tasks are processed concurrently. With live visualization of the tasks being processed, any bottleneck within the parallel workflow can be identified for further optimization.
:::

:::{tab-item} Visualization
`daspal` provides the option to choose from two approaches: a static representation based on `Matplotlib` and an interactive representation available for dynamic visualization of the data using `Holoview`. The resulting dynamic rendering can be applied directly on the lazy Dask-backed Datarray that is recomputed on the fly as the data representation changes with a zoomed window. This allow flexibility between obtaining production ready plots vs dynamic visualization for data exploration.
:::


::::


