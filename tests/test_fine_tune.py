# tests/test_fine_tune.py

import unittest
from src.fine_tune import prepare_training_data

class TestFineTune(unittest.TestCase):

    def test_prepare_training_data(self):
        data = prepare_training_data()
        self.assertIsInstance(data, list)  # Check if the data is a list
        self.assertGreater(len(data), 0)    # Ensure there is at least one entry

if __name__ == "__main__":
    unittest.main()