from configparser import ConfigParser
import logging
from .constants import *
from .strings import *
import sys
import numpy as np



def get_config_parser(filename = 'config.cfg'):
    config = ConfigParser(allow_no_value=True)
    config.read(filename)
    return config

def get_logger(config):
    formatter = logging.Formatter(logging.BASIC_FORMAT)   
    info_handler = logging.FileHandler(config[LOG_FILE])
    info_handler.setLevel(logging.DEBUG)
    info_handler.setFormatter(formatter)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(formatter)

    logger = logging.getLogger(name=DEEP_TRADING_AGENT)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(out_handler)
    
    return logger

def save_npy(obj, path, logger):
    np.save(path, obj)
    message = "  [*] saved at {}".format(path)
    logger.info(message)

def load_npy(path, logger):
    obj = np.load(path)
    message = "  [*] loaded from {}".format(path)
    logger.info(message)
    return obj