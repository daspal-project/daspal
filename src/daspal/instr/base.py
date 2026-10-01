"""
Base instrument interface definitions.
Defines the abstract instrument class used by all
instrument-specific implementations.
"""

from abc import ABC, abstractmethod
import datetime
import numpy as np
import xarray as xr

class Instr(ABC):
    """
    Abstract base class for all instruments.
    Provides the basic functions that are required 
    from all instruments for things to work.
    """
    ######## set values ########
    #file_length = set()
    #meta_exclude = set()

    #########################################################################
    ############################ FINDING DATA ###############################
    """functions used in io.finder""" 
    
    @abstractmethod
    def get_time_path(self, base_path: str, date_dir: str, datatype: str) -> str:
        """
        Return the folder where the files for this instrument are stored.
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_datetime(self, date_dir: str, filename: str) -> datetime.datetime:
        """
        Extract datetime of a file, possibly using date_dir if in path.
        """
        raise NotImplementedError

    @abstractmethod
    def sort_files(self, files: list) -> list:
        """
        Select files with the right extension ('hdf5' or 'h5') and sort them.
        """
        raise NotImplementedError
    
    #########################################################################
    ####################### READING DATA & METADATA #########################

    @abstractmethod
    def read_data_info(self, fname: str) -> dict:
        """
        Extracts header info and dataset shape for a single file.
        function used in in daspal.io.read
        """
        raise NotImplementedError
        
    @abstractmethod
    def read_data(self, fname: str, dist_indices: np.ndarray) -> np.ndarray:
        """
        Efficiently read a channel window from an HDF5 dataset.
        dist_sel can be a slice or np.ndarray
        function used in in daspal.io.read
        """
        raise NotImplementedError

    @abstractmethod    
    def metadata(self, fname: str, meta: dict) -> dict:
        """
        Interpret instrument metadata 
        function used in daspal.core
        """
        raise NotImplementedError

    #########################################################################
    ########################## PRE-PROCESSING DATA ##########################
    
    @abstractmethod        
    def process(self, xarr: xr.DataArray, 
        strain: bool, unwrap: bool, integrate: bool) -> xr.DataArray:
        """
        Pre-Process DataArray for strain conversion,
        phase unwrapping or integration when available
        """
        raise NotImplementedError


    #########################################################################
    ############################# SAVING DATA ###############################

    @abstractmethod 
    def format(self, xarr: xr.DataArray, **auxmeta):
        """
        Format the DataArray metadata in specific
        format related to instrument type
        """
        raise NotImplementedError