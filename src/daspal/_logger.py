"""
Library for setting up log messages
"""

import logging

def set_logger(level='warning'):
#def set_logger(level='info'):    
#def set_logger(level=None):
    """
    Returns the daspal logger.
    
    level: str, one of "debug", "info", "warning"
           None = silent (no output)
    """
    logger = logging.getLogger("daspal")

    if not logger.hasHandlers(): # avoids duplicate output
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if level is None:
        logger.setLevel(logging.CRITICAL + 1) # silent by default, higher than max level
    else:    
        log_level(level)

    return logger

def log_level(level):
    logger = logging.getLogger("daspal")
    level_map = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
    logger.setLevel(level_map.get(level.lower(), logging.INFO))

logger = set_logger()