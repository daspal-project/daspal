"""
APSensing instrument implementation.
"""

import os, re
import datetime
import h5py
import numpy as np
import xarray as xr

import daspal.instr.base as base
import daspal.instr.registry as registry
import daspal.io.h5fc as h5fc
from daspal.io import zarrfc

from daspal._logger import logger

# -------------------
# Instrument subclass
# -------------------
class apsensingInstr(base.Instr):
    """
    Defines instrument-specific constants and
    data formatting logic for APSensing DAS systems.
    """

    ######## set values ########
    file_length = 300 # 300s files for "raw" data
    meta_exclude = ['DAS'] # exclude data

    #########################################################################
    ############################ FINDING DATA ###############################
    """functions used in io.finder""" 
    
    def get_time_path(self, base_path, date_dir, datatype):
        # dxs instrument might not use datatype folder
        return os.path.join(base_path, date_dir)
        
    def extract_datetime(self, date_dir: str, filename: str) -> datetime.datetime:
        """
        Extract whole timestamp directly from filename.
        """
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2}\.\d{2}.\d{5})' # timestamp pattern to match in the filenames
        match = re.search(timestamp_pattern, filename)
        timestamp_str = match.group(0)  # Extract the matched timestamp
        # Replace "_" with space and "." in time with ":"
        timestamp_str = timestamp_str.replace("_", " ", 1)  # only replace first "_"
        date_part, time_part = timestamp_str.split(" ")
        # fix microseconds: pad to 6 digits
        date_time_str = f"{date_part} {time_part.replace('.', ':', 2)}"
    
        # Parse into datetime
        date_time = datetime.datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S.%f")
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
            t0 = np.datetime64(np.array(file['Timestamps']['DataTimestamps']).reshape(-1)[0].astype('datetime64[us]'), "ns")
            dt = np.timedelta64(round(1e9 / file['ProcessingServer']['DataRate'][0]), "ns")
            dx = file['ProcessingServer']['SpatialSampling'][0]
            d0 = np.array(file['Distances']['MeterPositions']).reshape(-1)[0]

            # dataset info
            dataset = file['DAS']
            dshape = dataset.shape
            dtype = dataset.dtype

        return {"t0": t0, "dt": dt, "dx": dx,  "d0": d0,
                "dshape": dshape, "dtype": dtype}

    def read_data(self, fname: str, dist_indices: np.ndarray) -> np.ndarray:
        """
        Efficiently read a channel window from an HDF5 dataset.
        dist_sel can be a slice or np.ndarray
        function used in in daspal.io.read
        """
        with h5py.File(fname, 'r') as file:
            dataset = file['DAS']
            return dataset[:, dist_indices]
    
    def metadata(self, fname, meta) -> dict:
        """
        Interpret APSensing metadata 
        function used in daspal.core
        """
        
        with h5py.File(fname, 'r') as file:
            mfields = h5fc.load_h5fields(file)#, skip=meta_exclude) # skip data
        
        m = dict(
            dimensions=['time', 'distance'],
            dimensionsU=['s', 'm'],
            dt=1/mfields['ProcessingServer']['DataRate'][0],
            dx=mfields['ProcessingServer']['SpatialSampling'][0],
            gaugeLength=mfields['ProcessingServer']['GaugeLength'][0],
            instrument='apsensing',
            name='',
            time=np.array(mfields['Timestamps']['DataTimestamps']).reshape(-1)[0],
            unit='',
            #AcquisitionDescription  = mfields['Acquisition']['AcquisitionDescription'],
            #MinimumFrequency = mfields['Acquisition']['MinimumFrequency'], # Hz
            #MaximumFrequency = mfields['Acquisition']['MaximumFrequency'], # Hz
            #PulseRate = mfields['Acquisition']['PulseRate'], # Hz
            #PulseWidth = mfields['Acquisition']['PulseWidth'] * 1e-9, # (in ns convert to seconds)
            #TriggeredMeasurement = mfields['Acquisition']['TriggeredMeasurement'],
            #OutputDataRate = mfields['Acquisition']['Raw[0]']['OutputDataRate'] # Hz
        )

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

        logger.warning("Pre-processing not available yet for apsensing data")
        
        return xarr

    #########################################################################
    ############################# SAVING DATA ###############################

    def format(self, xarr: xr.DataArray, **auxmeta):
        """
        Format to `apsensing` data format before saving
        function used in daspal.io.write
        """

        # make it appear only once
        logger.warning(f"Format to apsensing not available yet, "
            f"formating to daspal native `dasproc` format instead")

        # format metadata to `dasproc` standard
        ffiddict = registry.get_instrument("dasproc").format(xarr)
        
        return ffiddict