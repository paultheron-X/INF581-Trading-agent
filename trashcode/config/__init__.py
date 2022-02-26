from .config_master import *
from .constants import *
from .strings import *

config_p = get_config_parser('/Users/paultheron/Desktop/INF581 - Project/INF581-Trading-agent/config/config.cfg')
#config_p = get_config_parser('/Users/jeremie/Documents/02 -Scolarité/01 -Polytechnique/11 -Cours 3A/02 -P2/02 -INF581/Project/INF581-Trading-agent/config/config.cfg')
config = get_config(config_p)
logger = get_logger(config)