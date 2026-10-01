"""
daspal `dasproc` native format implementation.
"""

import os, re
import datetime
import h5py
import numpy as np
import xarray as xr

import daspal.instr.base as base
import daspal.io.h5fc as h5fc

from daspal._logger import logger

# -------------------
# Instrument subclass
# -------------------
class dasprocInstr(base.Instr):
    """
    Defines instrument-specific constants and
    data formatting logic for daspal processed data.
    """
    
    ######## set values ########
    meta_exclude = ['data']

    #########################################################################
    ############################ FINDING DATA ###############################
    """functions used in io.finder""" 
    
    def get_time_path(self, base_path, date_dir, datatype):
        # optodas has date_dir/datatype structure
        return os.path.join(base_path, date_dir, datatype)
        
    def extract_datetime(self, date_dir: str, filename: str) -> datetime.datetime:
        """
        Extract whole timestamp directly from path and filename.
        Flexibility of format
        """
        if date_dir is not None:
            time_str = os.path.splitext(filename)[0]  # '235954'
            t_str = f"{date_dir}T{time_str}"      # '20231111T235954'
            date_time = datetime.datetime.strptime(t_str, "%Y%m%dT%H%M%S")
        else:        
            timestamp_pattern = r'(\d{8}T\d{6}\.\d{6}Z)' # timestamp pattern to match in the filenames
            match = re.search(timestamp_pattern, filename)
            timestamp_str = match.group(0)  # Extract the matched timestamp
            timestamp_str = timestamp_str.replace('Z', '')
            date_time = datetime.datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S.%f')
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
            t0 = np.datetime64(round(file['time'][()] * 1e9), "ns")
            dt = np.timedelta64(round(1e9 * file["dt"][()]), "ns")
            dx = file["dx"][()]
            d0 = file["distance"][()]
            
            # dataset info
            dataset = file["data"]
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
            data = f['data']
            return data[:, dist_slice]
    
    def metadata(self, fname, meta) -> dict:
        """
        Interpret dasproc metadata 
        function used in daspal.core
        """
        with h5py.File(fname, 'r') as file:
            mfields = h5fc.load_h5fields(file)#, skip=meta_exclude) # skip data

        m = dict(mfields)

        if meta is not None:
           m.update(meta) 
            
        return m
            
    #########################################################################
    ########################## PROCESSING DATA ##############################

    def process(self,
        xarr: xr.DataArray, 
        strain: bool,
        unwrap: bool,
        integrate: bool
        ) -> xr.DataArray:

        logger.warning("Pre-processing not available yet for dasproc data")
        
        return xarr
        
    #########################################################################
    ############################# SAVING DATA ###############################

    def format(self, xarr: xr.DataArray, **auxmeta):
        """
        Format to daspal native `dasproc` format before saving
        function used in daspal.io.write and 
        """
        data_array = xarr.copy()

        attrs = data_array.attrs

        attrs.setdefault('project', 'unkown')
        attrs.setdefault('cable', 'unkown')
        attrs.setdefault('fibre', 'unkown')

        keys = ['project', 
                'fibre', 
                'cable',
                'experiment',
                'name',
                'unit',
                'dimensions', 
                'dimensionsU',
                'time',
                'distance', 
                'dt', 
                'dx',
                'gaugeLength',
                'instrument']
        
        m = {k: attrs[k] for k in keys if k in attrs}

        if not isinstance(attrs.get('instrument'), dict):
            m['instrument'] = {
                'type': attrs.get('instrument', 'unknown'),
                'parameters': {
                k: v for k, v in attrs.items() if k not in keys and k != 'instrument'
                }
            }
        else: # recreate order
            m['instrument'] = {
                'type': m['instrument']['type'],
                'parameters': m['instrument']['parameters']
            }

        m.update(auxmeta) 
            
        return m