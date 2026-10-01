import os
import datetime
from concurrent.futures import ThreadPoolExecutor

from daspal._logger import logger

class Finder:
    """
    Finder: Locate and organize DAS data files within a specified time window.

    This class scans a folder (or folder hierarchy) for data files corresponding
    to a given instrument and datatype. It can detect gaps in time coverage and
    return either a flat list of continuous files or a dictionary of contiguous blocks.

    Parameters
    ----------
    path : str
        Base directory where data files are stored.
    start : datetime.datetime
        Start time of the requested data window (inclusive).
    end : datetime.datetime
        End time of the requested data window (exclusive).
    instrument : object
        Instrument object with methods to extract timestamps and get file paths.
    datatype : str
        Type of data to load, e.g., 'dphi', or 'proc'.
    file_length : float, optional
        Duration of each file in seconds. If None, the instrument's default is used.
    check_gaps : bool, default=True
        If True, gaps in time coverage are detected and files are organized into blocks.

    Attributes
    ----------
    _path, _start, _end, _instrument, _datatype, _file_length, _check_gaps
        Stored parameters for internal use.

    Notes
    -----
    - `find_files()` is the main method to retrieve sorted files or blocks.
    - The class does **not load the data**; it only organizes file paths.
    - Gap detection is optional but recommended for accurate downstream processing.

    Example
    -------
    finder = Finder(
        path="/data/",
        start=datetime(2023,2,1,12,0),
        end=datetime(2023,2,1,14,0),
        instrument=optodas,
        datatype="dphi"
    )
    filepaths = finder.find_files()  # returns list or dict of blocks
    """

    def __init__(self, path, start, end, instrument, datatype, file_length=None, check_gaps=True):
        self._path = path
        self._start = start
        self._end = end
        self._instrument = instrument
        self._datatype = datatype
        self._check_gaps = check_gaps

       # Determine _file_length
        if file_length is not None:
            # Manual override always wins
            self._file_length = file_length
        elif (datatype is None or datatype.lower() == "dphi") and hasattr(instrument, "file_length"):
            # Use instrument default only if datatype is missing or 'dphi'
            self._file_length = instrument.file_length
        else:
            # Fallback: user must provide file_length
            raise ValueError(
                f"`file_length` is not defined for instrument '{type(instrument).__name__}'"
                f" with datatype '{datatype}'."
                f"Please provide it manually."
            )

    def find_files(self):

        files = []
        date_dirs = self._find_date_dirs()  # Identify date directories inside the defined path

        with ThreadPoolExecutor(max_workers=4) as executor:
            if date_dirs:
                for date_dir in date_dirs:
                    if not self._is_date_in_range(date_dir):
                        continue
                        
                    time_path = self._instrument.get_time_path(self._path, date_dir, self._datatype)
                    
                    with os.scandir(time_path) as time_files:
                        objs = executor.map(
                            self._check_time_file(date_dir),
                            (tf for tf in time_files if tf.is_file())
                        )
                        files.extend(f for f in objs if f)
            else:
                # scan base folder directly
                with os.scandir(self._path) as time_files:
                    objs = executor.map(
                        self._check_time_file(None),
                        (tf for tf in time_files if tf.is_file())
                    )
                    files.extend(f for f in objs if f)

        if not files:
            logger.warning(f"No files found for interval {self._start} - {self._end}.")
            return []
        else:
            files = self._instrument.sort_files(files)

        ###### detect and prepare filepaths for gaps handling
        if self._check_gaps:
            files = self._process_gaps(files)

        return files

        
    def _check_time_file(self, date_dir=None):
        """ Returns a callable to filter files in the time range """
        def func(time_file):
      
            date_time = self._instrument.extract_datetime(date_dir, time_file.name)
            
            datestart = self._start - datetime.timedelta(seconds=self._file_length + 1)
            if datestart < date_time < self._end:
                return time_file.path
            return None
        return func


    def _find_date_dirs(self):
        """ Returns list of date folders in path """
        date_dirs = []
        with os.scandir(self._path) as dirs:
            for dir_ in dirs:
                if dir_.is_dir():
                    try:
                        # only keep folders matching YYYYMMDD
                        datetime.datetime.strptime(dir_.name, "%Y%m%d")
                        date_dirs.append(dir_.name)
                    except ValueError:
                        continue
        return sorted(date_dirs)

    
    def _is_date_in_range(self, date_dir):
        #datedir = simpledas.str2datetime(date_dir, True)
        datedir = datetime.datetime.strptime(date_dir, "%Y%m%d")
        datestart = self._start - datetime.timedelta(seconds=self._file_length+1)
        if datestart.date() <= datedir.date() <= self._end.date():
            return True
        else:
            return False
            

    def _process_gaps(self, filepaths):
        """
        Process a list of filepaths, check for gaps, and generate blocks of continuous files if any gaps.
    
        Args:
            filepaths (list[str]): list of file paths
    
        Returns:
            dict or list: dictionary of blocks {block_1: [...], ...} or flat list if only one block
        """

        # expected file spacing in seconds
        step = self._file_length
        # minimum nb files per block
        min_block_size = 1
        # avoid gap detection for very tiny time diff (<dt) in file name
        # tolerance of 10kHz sampling rate (very unlikely to have higher sampling)
        # so any mismatch will be more in the file name rounding
        tolerance = 1e-5

        # extract timestamps using the instrument method
        files = []
        for path in filepaths:
            filename = os.path.basename(path)
            date_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
            try:
                datetime.datetime.strptime(date_folder, "%Y%m%d")
            except ValueError:
                date_folder = None

            t = self._instrument.extract_datetime(date_folder, filename)
            #t = self._instrument.extract_datetime_from_date_dir(date_folder, filename)
            files.append((t, path))

        # detect gap between requested start time and first file available
        if files[0][0] > self._start:
            gap = (files[0][0] - self._start).total_seconds()
            logger.warning(f"time range gap: START {self._start} -> first file {files[0][0]} (gap: {gap:.6f} s)")
        
        # check gaps while building continuous files blocks
        all_blocks = []
        block = [files[0][1]]
        prev_time = files[0][0]

        for time, path in files[1:]: # start on 2nd filepath for comparison with previous
            diff = (time - prev_time).total_seconds()
            gap = diff - step
            if abs(gap) > tolerance:
                logger.warning(f"Gap Detected: file starting {prev_time} -> next file starting {time} ({gap:.6f} s)")

            if abs(gap) <= tolerance:
                block.append(path)
            else:
                if len(block) >= min_block_size: # select block if more that min size
                    all_blocks.append(block)
                block = [path] # define potential next block

            prev_time = time

        # Final block
        if len(block) >= min_block_size:
            all_blocks.append(block)

        # detect gap between requested end time and last file available
        diff = (self._end - files[-1][0]).total_seconds()
        gap = diff - step
        if files[-1][0] < self._end and gap > tolerance:
            logger.warning(f"time range gap: last file {files[-1][0]} -> END {self._end} (gap: {gap:.6f} s)")

        # returning files blocks dic or flat list
        blocks_dict = {f"block_{i}": b for i, b in enumerate(all_blocks, 1)}

        if len(all_blocks) == 1 and len(all_blocks[0]) == len(filepaths):
            return filepaths
        else:
            return blocks_dict
