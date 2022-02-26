from .utils import get_config, get_config_parser


"""
Method: 
    > When you want to add a new variable to the config file
    
    1. Add it to the cfg file
    2. update the get_config in utils
    3. update yr program
    
"""

config_p = get_config_parser('dqn_base.cfg')
config = get_config(config_p)