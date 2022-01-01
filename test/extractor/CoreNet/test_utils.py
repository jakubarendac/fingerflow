# pylint: disable=no-self-use
import unittest
import numpy as np
import pandas as pd

from src.fingerflow.extractor.CoreNet import utils, constants


class PreprocessImageDataTest(unittest.TestCase):
    def test_correct_image_data(self):
        mock_input_data = np.zeros((100, 100))
        mock_output_data = np.expand_dims(np.zeros(constants.INPUT_SHAPE[:2]), axis=0)

        result_output_data = utils.preprocess_image_data(mock_input_data)

        np.testing.assert_array_equal(result_output_data, mock_output_data)


class GetDetectionDataTest(unittest.TestCase):
    def test_empty_model_output(self):
        mock_input_image = np.zeros((100, 100))
        mock_input_model_outputs = (
            np.array([np.zeros((100, 4), dtype=np.float32)]),
            np.zeros((1, 100), dtype=np.float32),
            np.zeros((1, 100), dtype=np.float32),
            np.array([0]))

        mock_output_data = pd.DataFrame(np.empty(0, dtype=np.dtype([('x1', np.int64),
                                                                    ('y1', np.int64),
                                                                    ('x2', np.int64),
                                                                    ('y2', np.int64),
                                                                    ('score', np.float32),
                                                                    ('w', np.int64),
                                                                    ('h', np.int64)])))

        result_output_data = utils.get_detection_data(mock_input_image, mock_input_model_outputs)

        pd.testing.assert_frame_equal(result_output_data, mock_output_data)

    def test_not_empty_model_output(self):
        mock_input_image = np.zeros((100, 100))

        mock_detection_bounding_boxes = np.zeros((100, 4))
        mock_detection_bounding_boxes[0, 0] = 0.2
        mock_detection_bounding_boxes[0, 1] = 0.2
        mock_detection_bounding_boxes[0, 2] = 0.4
        mock_detection_bounding_boxes[0, 3] = 0.4
        mock_detection_bounding_boxes[1, 0] = 0.6
        mock_detection_bounding_boxes[1, 1] = 0.6
        mock_detection_bounding_boxes[1, 2] = 0.7
        mock_detection_bounding_boxes[1, 3] = 0.7

        mock_detection_scores = np.zeros((1, 100), dtype=np.float32)
        np.put(mock_detection_scores, [0, 1], [0.5, 0.75])

        mock_input_model_outputs = (
            [mock_detection_bounding_boxes],
            mock_detection_scores,
            np.zeros((1, 100), dtype=np.float32),
            np.array([2]))

        mock_output_data = np.array(
            [(20, 20, 40, 40, 0.5, 20, 20),
             (60, 60, 70, 70, 0.75, 10, 10)],
            dtype=np.dtype(
                [('x1', np.int64),
                 ('y1', np.int64),
                 ('x2', np.int64),
                 ('y2', np.int64),
                 ('score', np.float32),
                 ('w', np.int64),
                 ('h', np.int64)]))

        mock_output_data = pd.DataFrame(mock_output_data)

        result_output_data = utils.get_detection_data(mock_input_image, mock_input_model_outputs)

        pd.testing.assert_frame_equal(result_output_data, mock_output_data)
