"""Sets the type of floating point used the algorithms.

Michel Bierlaire
Tue May 20 2025, 09:54:48
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


raw_value = os.getenv('BIOGEME_FLOAT_TYPE', '64')
if raw_value not in {'32', '64'}:
    logger.warning(f'Invalid float type: {raw_value}. Valid values: "32" or "64". Update the environment variable BIOGEME_FLOAT_TYPE. "64" is used by default')
    raw_value = '64'
FLOAT_TYPE = int(raw_value)
NUMPY_FLOAT = f'float{FLOAT_TYPE}'
# Only 32 and 64 are valid values here

MACHINE_EPSILON = np.finfo(NUMPY_FLOAT).eps
MAX_FLOAT = np.finfo(NUMPY_FLOAT).max

