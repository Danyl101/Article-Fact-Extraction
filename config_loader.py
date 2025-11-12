import yaml
import os

# Load config.yaml from project root
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Optionally, load once globally
config = load_config()
