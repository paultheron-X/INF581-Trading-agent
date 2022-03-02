from .utils import get_config, get_config_parser


"""
Method: 
    > When you want to add a new variable to the config file
    
    1. Add it to the cfg file
    2. update the get_config in utils
    3. update yr program
    
"""

config_dqn_base = get_config(get_config_parser('dqn_base.cfg'))

config_dqn_deepsense = get_config(get_config_parser('dqn_deepsense.cfg'))