#!/usr/bin/env python3
import logging

from tqdm import tqdm

# Get logger (def level: INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get console handler
# On log tout ce qui est possible ici (i.e >= level du logger)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Manage formatter
formatter = logging.Formatter(
    "[%(asctime)s] - %(name)s.%(funcName)s() - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(ch)

## Manage tqdm
# On créé une classe à utiliser à la place de celle de tqdm
# def level: INFO
class CustomTqdm(tqdm):
    level = logging.INFO

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def display(self, msg=None, pos=None):
        if logger.isEnabledFor(self.level):
            super().display(msg=msg, pos=pos)

    def close(self):
        if logger.isEnabledFor(self.level):
            super().close()

    @staticmethod
    def setLevel(level):
        CustomTqdm.level = level
