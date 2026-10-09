"""
OptoDAS instrument implementation.
"""

import os, re
import datetime
import h5py
import numpy as np
import xarray as xr
from functools import partial
import dask.array as da

import daspal.core as dp
import daspal.instr.base as base
import daspal.io.h5fc as h5fc
from daspal.io import zarrfc

import warnings
from daspal._logger import logger

# -------------------
# Instrument subclass
# -------------------
class optodasInstr(base.Instr):
    """
    Defines instrument-specific constants and
    data formatting logic for OptoDAS DAS systems.
    """

    ######## set values ########
    file_length = 10 # 10s files for "raw" data
    
    #########################################################################
    ############################ FINDING DATA ###############################
    """functions used in io.finder""" 
    
    def get_time_path(self, base_path, date_dir, datatype):
        # optodas has date_dir/datatype structure
        return os.path.join(base_path, date_dir, datatype)

    def extract_datetime(self, date_dir: str, filename: str) -> datetime.datetime:
        """
        Extract whole timestamp directly from path and filename.
        """
        if date_dir is not None:
            time_str = os.path.splitext(filename)[0]  # '235954'
            t_str = f"{date_dir}T{time_str}"      # '20231111T235954'
            date_time = datetime.datetime.strptime(t_str, "%Y%m%dT%H%M%S")
        else:        
            raise ValueError("Cannot determine full datetime without date folder")
        return date_time
    
    def sort_files(self, files):
        # only keep files that end with end with the '.hdf5' and sort them
        files = [f for f in files if f.endswith('.hdf5')]
        return sorted(files)

    #########################################################################
    ####################### READING DATA & METADATA #########################
    
    def read_data_info(self, fname: str) -> dict:
        """
        Extracts header info and dataset shape for a single file.
        function used in in .io.read
        """
        with h5py.File(fname, 'r') as file:
            header = file['header']
            t0 = np.datetime64(round(header["time"][()] * 1e9), "ns")
            dt = np.timedelta64(round(1e9 * header["dt"][()]), "ns")
            dx = header["dx"][()] * np.median(np.diff(header["channels"]))
            d0 = header["channels"][()][0] * header["dx"][()] # defines distance zero if data was cut

            # dataset info
            dataset = file['data']
            dshape = dataset.shape
            dtype = dataset.dtype

        return {"t0": t0, "dt": dt, "dx": dx, "d0": d0,
                "dshape": dshape, "dtype": dtype}

    def read_data(self, fname: str, dist_slice: slice) -> np.ndarray:
        """
        Efficiently read a channel window from an HDF5 dataset.
        dist_sel can be a slice or np.ndarray
        function used in in .io.read
        """
        with h5py.File(fname, 'r') as f:
            dataset = f['data']
            return dataset[:, dist_slice]

    def metadata(self, fname, meta) -> dict:
        """
        Interpret OptoDAS metadata 
        function used in daspal.core
        """

        with h5py.File(fname, 'r') as file:
            mfields = h5fc.load_h5fields(file, skip=['data']) # skip data

        # selecting only metadata needed for DAS data interpretation
        m = dict(mfields["header"])

        m = {
            k: v for k, v in m.items() if k not in {'channels', 'dimensionRanges', 'dimensionSizes',  # info already in xarray
                                                   'missingSamples', 'numberOfAuxData',  # optodas single file specific
                                                   'phiOffs', 'phiOffsStartTime'}
        }

        m['dimensions']=m['dimensionNames']
        m['dimensionsU']=m['dimensionUnits']
        del m['dimensionNames'], m['dimensionUnits']
        m['instrument']='optodas'
        m['dxch']=m['dx']
        if m['name'] == 'Phase rate per distance':
            m['name']='phase rate per distance'

        if meta is not None:
            m.update(meta) 
        m = dict(sorted(m.items())) # sort metadata

        return m

    #########################################################################
    ########################## PRE-PROCESSING DATA ##########################

    def process(self,
        xarr: xr.DataArray, 
        strain: bool,
        unwrap: bool,
        integrate: bool
        ) -> xr.DataArray:

        """
        Apply processing to OptoDAS data.
        function used in .process

        Options:
        strain: Convert raw phase rate data to nanostrain rate.
           - Only valid if the input data is in time-differentiated phase units.
        unwrap: Spatial unwrapping along the fiber.
           - Uses a strain-rate threshold (`wrapStep`) derived from spatialUnwrRange and sensitivity.
           - Ensures continuity along the fiber and avoids artificial jumps from phase wrapping.
        integrate: Integrates strain rate to obtain cumulative strain.
           - option to use spatial unwrapping before to avoid propagating wrapped artifacts.

        Notes:
        Processing approach adapted from simpleDAS (https://github.com/ASN-Norway/simpleDAS),
        including the parameters described in simpleDAS/src/simpledas/simpleDASreader.py.
        """

        dataarray = xarr.copy()
        arr = dataarray.data
        shape = dataarray.shape
        t_axis = dataarray.dims.index('time')
        d_axis = dataarray.dims.index('distance')
        m = dataarray.attrs

        # --- convert to nanostrain (or not) ---
        
        ### setup metadata accordingly
        if strain:
            try:
                sensitivity = np.array(m["instrument"]["parameters"]["sensitivities"])[0, :]
            except KeyError:
                sensitivity = np.array(m["instrument"]["parameters"]["sensitivities"])
            except IndexError:
                print("SensitivitySelect index 0 not found in file.")
                raise

            unit_out = "nstrain/s"
            m["name"] = "nstrain rate"
            # sensitivity applied, output sensitivity set to 1
            sensitivities_out = np.ones((1, 1), dtype=np.float32)
            sensitivityUnits_out = [""]
        
        else:
            unit_out = m["unit"]  # rad/(s*m)
            sensitivity = 1.0
            sensitivities_out = m["instrument"]["parameters"]["sensitivities"]
            sensitivityUnits_out = m["instrument"]["parameters"]["sensitivityUnits"]
            
        ### data scaling based on metadata
        if not isinstance(sensitivity, np.ndarray) or sensitivity.size == 1:  # singelton
            if sensitivity != 0.0:
                scale = np.float32(m["instrument"]["parameters"]["dataScale"] / sensitivity)
            else:
                raise ValueError("Sensitivity value set to zero consider `strain` set to False to avoid error")
        else:
            logger.warning("Array with different sensitivity each channel")
            
            if shape[d_axis] != len(sensitivity):
                raise ValueError("Number of channels not matching size of sensitivity array")
            if np.any(sensitivity != 0.0):
                sensitivity = np.atleast_2d(sensitivity)
                scale = np.array(
                    m["instrument"]["parameters"]["dataScale"] / sensitivity, dtype=np.float32)
            else:
                raise ValueError("Sensitivity value set to zero consider `strain` set to False to avoid error")

        # convert data values to nano strain
        # keeping original data type (int16 or float32)
        arr_float = arr.astype(np.float32) * scale * 1e9
        arr = np.round(arr_float).astype(arr.dtype)

        # --- unwrap/ integrate options ---
        if (unwrap or integrate) and m["instrument"]["parameters"]["dataType"] == 2:
            unwrap, integrate = (False,) * 2
            warnings.warn("Use `unwrap` or `integrate` only with time differentiated phase data",
                          UserWarning)
        if unwrap and not strain:
            unwrap = False
            warnings.warn("Use `unwrap` on strain rate data", UserWarning)
            
        ##############
        ### unwrapping
        if unwrap and m["instrument"]["parameters"]["spatialUnwrRange"]: ########### still to be tested
            
            def data_unwrap(data: np.ndarray, wrapStep: float, axis: int = -1) -> np.ndarray:
                """
                ASN SimpleDAS approach
                Unwrapping based on strain rate threshold (wrapStep) for physical meaning.
                scale = 2π / wrapStep allows numpy.unwrap to operate in 2π range.
                """
                scale = 2 * np.pi / wrapStep
                return (np.unwrap(data * scale, axis=axis) / scale).astype(data.dtype)
            
            #wrapStep=m["spatialUnwrRange"] / sensitivity
            # redefining "spatialUnwrRange" in metadata from simpledas suggested value
            m["instrument"]["parameters"]["spatialUnwrRange"] = 8 * np.pi / m["dt"] / m["gaugeLength"] # -> rad/(s*m)
            wrapStep=m["instrument"]["parameters"]["spatialUnwrRange"] / sensitivity # -> strain/s

            if isinstance(arr, da.Array): # Dask version
                wrapped_unwrap = partial(data_unwrap, wrapStep=wrapStep, axis=d_axis)
                arr = arr.map_blocks(wrapped_unwrap, dtype=arr.dtype)  # Dask version

            else:
                arr = data_unwrap(arr, wrapStep=wrapStep, axis=d_axis)
        
        ##################
        ### integrate data
        if integrate:
            if not any([u == "s" for u in re.findall(r"[\w']+", unit_out)]) or m["instrument"]["parameters"]["dataType"] == 2:
                logger.warning(f"Integration skipped, data in unit {unit_out} cannot be integrated")
            else:
                unit_new = unit_out
                # handle "/s" units
                if unit_out.endswith("/s"):
                    unit_new = unit_out[: -len("/s")]
                # handle "rad/(s*m)" unit
                elif unit_out.startswith("rad/(s*") and unit_out.endswith(")"):
                    unit_new = unit_out.replace("(s*", "").replace(")", "")
                
                # integrate array
                dt = m["dt"]
                if isinstance(arr, da.Array): # Dask version
                    arr = da.cumsum(arr, axis=0) * dt
                else:
                    arr = np.cumsum(arr, axis=0) * dt
                
                unit_out = unit_new

                m["name"] = m["name"].replace(" rate", "")
                m["instrument"]["parameters"]["dataType"] = 2

        m.update(unit=unit_out)
        m["instrument"]["parameters"].update(
            sensitivities=sensitivities_out,
            sensitivityUnits=sensitivityUnits_out
        )

        # make metadata zarr compatible again
        m = zarrfc.meta_comp(m)
        
        return xr.DataArray(arr, dims=dataarray.dims, coords=dataarray.coords, attrs=m)

    #########################################################################
    ############################# SAVING DATA ###############################
    
    def format(self, xarr: xr.DataArray, **auxmeta):
        """
        Format to `optodas` data format before saving
        function used in .io.write
        """
        data_array = xarr.copy()
        shape = data_array.shape
        t_axis = data_array.dims.index('time')
        d_axis = data_array.dims.index('distance')

        m = data_array.attrs
        
        # update dx back to original optodas value
        m['dx'] = m['instrument']['parameters']['dxch']
        del m['distance'], m['instrument']['parameters']['dxch']

        # flatten keys in dict tree
        m = dp.flatten_meta(m)
        m['dimensionNames']=m['dimensions']
        m['dimensionUnits']=m['dimensionsU']
        del m['dimensions'], m['dimensionsU'],m['instrument']
        
        # creating optodas specific keys
        channels = data_array.coords['distance'].values / m['dx']
        dimensionRanges = {
           f'dimension{t_axis}': {
                'min': 0,
                'max': shape[t_axis] - 1,
                'size': shape[t_axis],
                'unitScale': m["dt"]
            },
            f'dimension{d_axis}': {
                'min': 0,
                'max': shape[d_axis] - 1,
                'size': shape[d_axis],
                'unitScale': m["dx"]
            }
        }
        dimensionSizes = [shape[t_axis], shape[d_axis]]

        # creating "header"
        header = dict(
            channels=channels,
            dimensionRanges=dimensionRanges,
            dimensionSizes=dimensionSizes,
        )
        header.update(m)
        header.update(auxmeta) 

        ffiddict = dict(header=dict())
        ffiddict["header"].update(header)

        ffiddict["header"] = dict(sorted(ffiddict["header"].items())) # sort metadata

        return ffiddict
