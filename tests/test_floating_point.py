import unittest
import os
import importlib
import numpy as np

import biogeme_optimization.floating_point as floating_point

class TestFloatingPointConfig(unittest.TestCase):

    def setUp(self):
        # Backup current environment variable
        self.original_env = os.environ.get('BIOGEME_FLOAT_TYPE')

    def tearDown(self):
        # Restore original environment variable
        if self.original_env is not None:
            os.environ['BIOGEME_FLOAT_TYPE'] = self.original_env
        else:
            os.environ.pop('BIOGEME_FLOAT_TYPE', None)

    def test_valid_float_type_32(self):
        os.environ['BIOGEME_FLOAT_TYPE'] = '32'
        importlib.reload(floating_point)
        self.assertEqual(floating_point.FLOAT_TYPE, 32)
        self.assertEqual(floating_point.NUMPY_FLOAT, 'float32')
        self.assertAlmostEqual(floating_point.MACHINE_EPSILON, np.finfo(np.float32).eps)
        self.assertAlmostEqual(floating_point.MAX_FLOAT, np.finfo(np.float32).max)

    def test_valid_float_type_64(self):
        os.environ['BIOGEME_FLOAT_TYPE'] = '64'
        importlib.reload(floating_point)
        self.assertEqual(floating_point.FLOAT_TYPE, 64)
        self.assertEqual(floating_point.NUMPY_FLOAT, 'float64')
        self.assertAlmostEqual(floating_point.MACHINE_EPSILON, np.finfo(np.float64).eps)
        self.assertAlmostEqual(floating_point.MAX_FLOAT, np.finfo(np.float64).max)

    def test_invalid_float_type_fallback(self):
        os.environ['BIOGEME_FLOAT_TYPE'] = 'invalid'
        importlib.reload(floating_point)
        self.assertEqual(floating_point.FLOAT_TYPE, 64)
        self.assertEqual(floating_point.NUMPY_FLOAT, 'float64')

if __name__ == '__main__':
    unittest.main()