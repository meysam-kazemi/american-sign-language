import configparser
import os

def load_config(filename='config.ini'):
    """
    Loads configuration settings from an INI file.

    Args:
        filename (str): The path to the configuration file.

    Returns:
        configparser.ConfigParser: The configuration object, or None if the file is not found.
    """
    if not os.path.exists(filename):
        print(f"Error: Configuration file '{filename}' not found.")
        return None
        
    config = configparser.ConfigParser()
    config.read(filename)
    return config
