# pylint: disable=no-self-use
import unittest
import numpy as np
import pandas as pd

from src.fingerflow.extractor.ClassifyNet import utils


class FormatClassifiedDataTest(unittest.TestCase):
    def test_empty_numpy_data(self):
        mock_output_data = pd.DataFrame([], columns=['x', 'y', 'angle', 'score', 'class'])

        result_output_data = utils.format_classified_data(np.array([]))

        pd.testing.assert_frame_equal(result_output_data, mock_output_data)

    def test_not_empty_numpy_data(self):
        mock_input_data = np.array([[1, 1, 1, 0.5, 1], [2, 2, 2, 0.2, 2]])
        mock_output_data = pd.DataFrame(mock_input_data, columns=[
                                        'x', 'y', 'angle', 'score', 'class'])

        result_output_data = utils.format_classified_data(mock_input_data)

        pd.testing.assert_frame_equal(result_output_data, mock_output_data)
