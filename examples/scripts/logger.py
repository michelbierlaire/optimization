"""File logger.py

:author: Michel Bierlaire
:date: Mon Jul  3 11:46:47 2023

File setting the logger
"""

import logging

logger = logging.getLogger('biogeme_optimization')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(message)s ')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
