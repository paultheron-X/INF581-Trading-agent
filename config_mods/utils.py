from configparser import ConfigParser
from os.path import join
import os
import json


def get_config_parser(filename='dqn_base.cfg'):
    config = ConfigParser(allow_no_value=True)
    config.read(os.getcwd() + '/config_mods/' + filename)
    return config


def fill_config_with(config, config_parser, modifier, section, option):
    if config_parser.has_section(section) and config_parser.has_option(section, option):
        config[option.lower()] = modifier(config_parser.get(section, option))
    return config


def get_config(config_parser):
    config = {}

    config['df_path'] = os.getcwd() + '/gym_trading_btc/datasets/data/' + \
        config_parser.get('dataset', 'DF_NAME')

    fill_config_with(config, config_parser, int, 'dataset', 'NUM_FEATURES')
    fill_config_with(config, config_parser, float, 'dataset', 'TRAIN_TEST_SPLIT')
    
    config['save_path'] = os.getcwd() + config_parser.get('agent', 'SAVE_PATH')
    
    fill_config_with(config, config_parser, int, 'agent', 'SAVE')
    fill_config_with(config, config_parser, int, 'agent', 'LOAD')

    fill_config_with(config, config_parser, int, 'agent', 'NUM_ACTIONS')
    fill_config_with(config, config_parser, int, 'agent', 'WINDOW_SIZE')
    fill_config_with(config, config_parser, int, 'agent', 'FRAME_LEN')
    fill_config_with(config, config_parser, int, 'agent', 'NUM_EPISODE')
    fill_config_with(config, config_parser, int, 'agent', 'MAX_MEM_SIZE')

    fill_config_with(config, config_parser, float, 'agent', 'EXPLORATION_RATE')
    fill_config_with(config, config_parser, float, 'agent', 'EXPLORATION_DECAY')
    fill_config_with(config, config_parser, float, 'agent', 'EXPLORATION_MIN')

    fill_config_with(config, config_parser, str, 'classifier', 'OBJECTIVE')
    fill_config_with(config, config_parser, str, 'classifier', 'MODEL')
    fill_config_with(config, config_parser, int, 'classifier', 'MAX_ITER')

    fill_config_with(config, config_parser, int, 'dqn', 'BATCH_SIZE')
    fill_config_with(config, config_parser, int, 'dqn', 'REPLACE_TARGET')

    fill_config_with(config, config_parser, float, 'dqn', 'LR')
    fill_config_with(config, config_parser, float, 'dqn', 'GAMMA')

    fill_config_with(config, config_parser, json.loads, 'dqn', 'HIDDEN_SIZE')

    fill_config_with(config, config_parser, float, 'dropout', 'LINEAR_DROPOUT')
    fill_config_with(config, config_parser, float, 'dropout', 'CONV_DROPOUT')
    fill_config_with(config, config_parser, float, 'dropout', 'GRU_DROPOUT')

    fill_config_with(config, config_parser, json.loads,
                     'convolution', 'FILTER_SIZES')
    fill_config_with(config, config_parser, json.loads,
                     'convolution', 'KERNEL_SIZES')

    fill_config_with(config, config_parser, str, 'convolution', 'PADDING')
    fill_config_with(config, config_parser, int, 'convolution', 'STRIDE')

    fill_config_with(config, config_parser, int, 'gru', 'GRU_CELL_SIZE')
    fill_config_with(config, config_parser, int, 'gru', 'GRU_NUM_CELLS')

    fill_config_with(config, config_parser, int, 'print', 'TRAINING_STATE')
    
    if ('train_test_split' not in config):
        raise ValueError('No train test repartition is given')
    if ('hidden_size' in config) and len(config['hidden_size']) <= 0:
        raise ValueError('Hidden size parameter is empty')
    if ('stride' in config and 'deepsense' in config) and (config['window_size'] % config['stride'] != 0) and config['deepsense']:
        raise ValueError('Stride and Window params not compatible')
    if ('gru_cell_size' in config and 'deepsense' in config and 'hidden_size' in config) and (config['gru_cell_size'] != config['hidden_size'][0]) and config['deepsense']:
        raise ValueError('gru cell and hidden size not compatible')

    return config
