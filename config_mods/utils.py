from configparser import ConfigParser
from os.path import join
import os, sys


def get_config_parser(filename = 'dqn_base.cfg'):
    config = ConfigParser(allow_no_value=True)
    config.read(os.getcwd() + '/config_mods/' + filename)
    return config

def get_config(config_parser):
    config = {}
    
    config['df_path'] = os.getcwd() + '/gym_trading_btc/gym_anytrading/datasets/data/' + config_parser.get('dataset', 'DF_NAME')
    config['num_features'] = int(config_parser.get('dataset', 'NUM_FEATURES'))
    
    config['num_actions'] = int(config_parser.get('agent', 'NUM_ACTIONS'))
    config['window_size'] = int(config_parser.get('agent', 'WINDOW_SIZE'))
    config['frame_len'] = int(config_parser.get('agent', 'FRAME_LEN'))
    config['num_episode'] = int(config_parser.get('agent', 'NUM_EPISODE'))
    
    config['batch_size'] = int(config_parser.get('dqn', 'BATCH_SIZE'))
    config['replace_target'] = int(config_parser.get('dqn', 'REPLACE_TARGET'))
    
    config['dropout'] = float(config_parser.get('dqn', 'DROPOUT'))
    config['lr'] = float(config_parser.get('dqn', 'LR'))
    config['gamma'] = float(config_parser.get('dqn', 'GAMMA'))
    config['hidden_size'] = int(config_parser.get('dqn', 'HIDDEN_SIZE')) #json.loads
    
    config['max_mem_size'] = int(config_parser.get('dqn', 'MAX_MEM_SIZE'))
    
    config['exploration_rate'] = float(config_parser.get('dqn', 'EXPLORATION_RATE'))
    config['exploration_decay'] = float(config_parser.get('dqn', 'EXPLORATION_DECAY'))
    config['exploration_min'] = float(config_parser.get('dqn', 'EXPLORATION_MIN'))
    
    config['training_state'] = int(config_parser.get('print', 'TRAINING_STATE'))
    
    return config