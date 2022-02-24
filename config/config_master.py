from configparser import ConfigParser
import logging
from .constants import *
from .strings import *
import sys
import numpy as np
import json
from os.path import join


def add_parent_dir(parent_dir, path):
    return join(parent_dir, path)

def get_config(config_parser):
    config = {}
    #Global
    config[PARENT_DIR] = config_parser.get(GLOBAL, PARENT_DIR)

    #Logging
    config[LOG_FILE] = add_parent_dir(config[PARENT_DIR], config_parser.get(LOGGING, LOG_FILE))
    config[SAVE_DIR] = add_parent_dir(config[PARENT_DIR], config_parser.get(LOGGING, SAVE_DIR))
    config[TENSORBOARD_LOG_DIR] = add_parent_dir(config[PARENT_DIR], 
                                                    config_parser.get(LOGGING, TENSORBOARD_LOG_DIR))

    #Preprocessing Dataset
    config[DATASET_PATH] = add_parent_dir(config[PARENT_DIR], 
                                            config_parser.get(PREPROCESSING, DATASET_PATH))

    #Dataset Parameters
    config[BATCH_SIZE] = int(config_parser.get(DATASET, BATCH_SIZE))
    config[HISTORY_LENGTH] = int(config_parser.get(DATASET, HISTORY_LENGTH))
    config[HORIZON] = int(config_parser.get(DATASET, HORIZON))
    config[MEMORY_SIZE] = int(config_parser.get(DATASET, MEMORY_SIZE))
    config[NUM_ACTIONS] = int(config_parser.get(DATASET, NUM_ACTIONS))
    config[NUM_CHANNELS] = int(config_parser.get(DATASET, NUM_CHANNELS))
    config[SPLIT_SIZE] = int(config_parser.get(DATASET, SPLIT_SIZE))
    config[WINDOW_SIZE] = int(config_parser.get(DATASET, WINDOW_SIZE))

    #Dropout Layer Parameters
    config[CONV_KEEP_PROB] = float(config_parser.get(DROPOUT, CONV_KEEP_PROB))
    config[DENSE_KEEP_PROB] = float(config_parser.get(DROPOUT, DENSE_KEEP_PROB))
    config[GRU_KEEP_PROB] = float(config_parser.get(DROPOUT, GRU_KEEP_PROB))

    #Convolution Layer Parameters
    config[FILTER_SIZES] = json.loads(config_parser.get(CONVOLUTION, FILTER_SIZES))
    config[KERNEL_SIZES] = json.loads(config_parser.get(CONVOLUTION, KERNEL_SIZES))
    config[PADDING] = config_parser.get(CONVOLUTION, PADDING)

    #GRUCell Parameters
    config[GRU_CELL_SIZE] = int(config_parser.get(GRU, GRU_CELL_SIZE))
    config[GRU_NUM_CELLS] = int(config_parser.get(GRU, GRU_NUM_CELLS))

    #FullyConnected Layer Parameters
    config[DENSE_LAYER_SIZES] = json.loads(config_parser.get(DENSE, DENSE_LAYER_SIZES))

    if (config[SPLIT_SIZE] * config[WINDOW_SIZE] != config[HISTORY_LENGTH]):
        raise ValueError(DIMENSION_MISMATCH_HISTORY_LENGTH)

    return config


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