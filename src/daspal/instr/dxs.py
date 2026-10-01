"""
DxS instrument implementation.
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
class dxsInstr(base.Instr):
    """
    Defines instrument-specific constants and
    data formatting logic for DxS DAS systems.
    """

    ######## set values ########
    file_length = 5 # 5s files for "raw" data
    meta_exclude = ['Acquisition/Raw[0]/RawData'] # exclude data
    
    #########################################################################
    ############################ FINDING DATA ###############################
    """functions used in io.finder""" 
    
    def get_time_path(self, base_path, date_dir, datatype):
        # DxS instrument might not use datatype folder
        return os.path.join(base_path, date_dir)
        
    def extract_datetime(self, date_dir: str, filename: str) -> datetime.datetime:
        """
        Extract whole timestamp directly from filename.
        """
        timestamp_pattern = r'(\d{8}T\d{6}\.\d{6}Z)' # timestamp pattern to match in the filenames
        match = re.search(timestamp_pattern, filename)
        timestamp_str = match.group(0)  # Extract the matched timestamp
        timestamp_str = timestamp_str.replace('Z', '')
        date_time = datetime.datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S.%f')
        return date_time

    def sort_files(self, files):
        # only keep files that end with end with the '.h5' and sort them
        files = [f for f in files if f.endswith('.h5')]
        return sorted(files)
    
    #########################################################################
    ####################### READING DATA & METADATA #########################
    
    def read_data_info(self, fname: str) -> dict:
        """
        Extracts header info and dataset shape for a single file.
        function used in in .io.read
        """
        with h5py.File(fname, 'r') as file:
            raw0 = file['Acquisition/Raw[0]']
            t0 = np.datetime64(raw0['RawDataTime'][0].astype('datetime64[us]'), "ns")
            dt = np.timedelta64(round(1e9 / raw0.attrs['OutputDataRate'][()]), "ns")
            dx = file['Acquisition'].attrs['SpatialSamplingInterval'][()]
            d0 = file['/Acquisition/FacilityCalibration[0]/Calibration[0]/LocusDepthPoint']['OpticalPathDistance'][0]

            # dataset info
            dataset = raw0['RawData']
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
        with h5py.File(fname, 'r') as file:
            dataset = file['Acquisition/Raw[0]/RawData']
            return dataset[:, dist_slice]

    def metadata(self, fname, meta) -> dict:
        """
        Interpret DXS metadata 
        function used in daspal.core
        """

        with h5py.File(fname, 'r') as file:
            mfields = h5fc.load_h5fields(file)#, skip=meta_exclude) # skip data
        
        # to be tuned
        if mfields['Acquisition']['Raw[0]']['RawDataUnit'] == 'unitless':
            name = 'phase'
            unit = 'rad'
        else:
            name = ''
            unit_phase = mfields['Acquisition']['Raw[0]']['RawDataUnit']
       
        m = dict(
            dimensions=['time', 'distance'],
            dimensionsU=['s', 'm'],
            dt=1 / mfields['Acquisition']['Raw[0]']['OutputDataRate'][()],
            dx=mfields['Acquisition']['SpatialSamplingInterval'][()],
            gaugeLength=mfields['Acquisition']['GaugeLength'][()],
            instrument='dxs',
            name=name,
            time=mfields['Acquisition']['Raw[0]']['RawDataTime'][0],
            unit=unit,
            AcquisitionDescription  = mfields['Acquisition']['AcquisitionDescription'],
            MinimumFrequency = mfields['Acquisition']['MinimumFrequency'], # Hz
            MaximumFrequency = mfields['Acquisition']['MaximumFrequency'], # Hz
            PulseRate = mfields['Acquisition']['PulseRate'], # Hz
            PulseWidth = mfields['Acquisition']['PulseWidth'] * 1e-9, # (in ns convert to seconds)
            TriggeredMeasurement = mfields['Acquisition']['TriggeredMeasurement'],
            OutputDataRate = mfields['Acquisition']['Raw[0]']['OutputDataRate'] # Hz
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

        """
        Apply processing to DxS data.
        function used in .process

        Options:
        strain: Convert raw phase data to strain rate.
           - Only valid if the input data is in time-differentiated phase units.
        unwrap: Spatial unwrapping along the fiber.
            NOT implemented yet for DxS data
        integrate: Integrates strain rate to obtain cumulative strain.
        """

        dataarray = xarr.copy()
        arr = dataarray.data
        shape = dataarray.shape
        t_axis = dataarray.dims.index('time')
        d_axis = dataarray.dims.index('distance')
        m = dataarray.attrs

        # --- convert to strain (or not) ---

        if strain:
            #    Strain=(Lambda*phase)/(4*pi*n*E).
            Lambda = 1550 # in nm
            n=1.469 # value for fiber refraction index
            E=0.78
            scale = Lambda/(4*np.pi*n*E)
       
            unit_out = "nstrain"
            m["name"] = m["name"].replace("phase", "nstrain")

            arr = arr * scale # scale data values based on sensitivity setup
        
        ##############
        ### unwrapping
        if unwrap:
            unwrap = False
            logger.warning("`unwrap` not implemented for DXS", UserWarning)
        
        ##################
        ### integrate data
        if integrate:
            integrate = False
            logger.warning("`integrate` not implemented for DXS", UserWarning)


        m.update(unit=unit_out)
   
        return xr.DataArray(arr, dims=dataarray.dims, coords=dataarray.coords, attrs=m)


    #########################################################################
    ############################# SAVING DATA ###############################

    def format(self, xarr: xr.DataArray, **auxmeta):
        """
        Format to `DxS` data format before saving
        function used in daspal.io.write
        """

        # make it appear only once
        logger.warning(f"Format to DxS not available yet, "
            f"formating to daspal native `dasproc` format instead")

        # format metadata to `dasproc` standard
        ffiddict = registry.get_instrument("dasproc").format(xarr)
        
        return ffiddict