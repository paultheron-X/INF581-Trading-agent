from configparser import ConfigParser


def get_config_parser(filename = 'config.cfg'):
    config = ConfigParser(allow_no_value=True)
    config.read(filename)
    return config
