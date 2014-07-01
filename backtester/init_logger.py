import logging.config
import yaml

def setup():
    with open('../conf/logging.yaml') as f:
        config = yaml.load(f)
        logging.config.dictConfig(config)