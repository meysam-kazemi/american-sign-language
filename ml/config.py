"""Configuration loading."""
import os
import configparser

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")


def load_config(filename=_DEFAULT_PATH):
    """Load settings from an INI file. Raises FileNotFoundError if missing."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")
    config = configparser.ConfigParser()
    config.read(filename)
    return config
