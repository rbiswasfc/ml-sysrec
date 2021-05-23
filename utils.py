import json


def load_config():
    """
    load config for current project

    :return: config dict
    :rtype: dict
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    return config
