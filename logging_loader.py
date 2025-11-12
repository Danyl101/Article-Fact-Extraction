import logging.config
import yaml

# Load logging before creating Flask app
with open("logging.yaml", "r") as f:
    log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)

