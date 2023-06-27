"""Script to run unit test on sensitivity analysis script.
By Nikolaus Ricker"""

# External Libraries
import numpy as np
import unittest

# Local Libraries
import sys
sys.path.append('')

# Internal Libraries
from mission import sensitivity_analysis


class TestSensitivity(unittest.TestCase):
    """Class to test sensitivity functions"""

    def test_size(self):
        """Test if the score matrix has two rows and the same number of columns as the weight matrix."""
        score = np.array([[1], 
                          [2]])
        weight = [range(1,7), 
                  range(1,7)]
        with self.assertRaises(ValueError):
            sensitivity_analysis(score, weight)

    def test_win(self):
        """Test if the winning is correct"""
        score = np.array([[1,2],
                          [2,1]])
        weight = [range(2,4),
                  range(2,4)]
        win = sensitivity_analysis(score, weight)
        self.assertEqual(win[0], 1)
        self.assertEqual(win[1], 1)
        self.assertEqual(win[2], 2)

if __name__ == "__main__":
    unittest.main()