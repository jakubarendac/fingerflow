# pylint: disable=no-self-use
import unittest
import numpy as np
import tensorflow as tf

from src.fingerflow.matcher.VerifyNet import utils

tf.config.set_visible_devices([], 'GPU')


class EuclideanDistanceTest(unittest.TestCase):
    def test_calculate_distance_to_same_point(self):
        tensor_1 = tf.constant([[10.0, 10.0]])
        tensor_2 = tf.constant([[10.0, 10.0]])

        result_output = utils.euclidean_distance([tensor_1, tensor_2])

        result_mock = tf.constant([[0.00031622776]])

        [[are_results_equal]] = tf.math.equal(result_output, result_mock).numpy()

        self.assertTrue(are_results_equal)

    def test_calculate_distance_to_different_point(self):
        tensor_1 = tf.constant([[10.0, 10.0]])
        tensor_2 = tf.constant([[20.0, 20.0]])

        result_output = utils.euclidean_distance([tensor_1, tensor_2])

        result_mock = tf.constant([[14.142136]])

        [[are_results_equal]] = tf.math.equal(result_output, result_mock).numpy()

        self.assertTrue(are_results_equal)


class GetInputShapeTest(unittest.TestCase):
    def test_input_shape_with_currently_set_minutiae_features(self):
        mock_precision = 20
        mock_minutiae_features = 9

        result_output = utils.get_input_shape(mock_precision)

        result_mock = (mock_precision, mock_minutiae_features, 1)

        self.assertTupleEqual(result_output, result_mock)


class EnhanceMinutiaePointsTest(unittest.TestCase):
    def test_empty_minutiae_input(self):
        mock_minutiae = np.array([])

        result_output = utils.enhance_minutiae_points(mock_minutiae)

        result_mock = np.array([])

        np.testing.assert_array_equal(result_output, result_mock)

    def test_correct_input_with_current_setup(self):
        mock_minutiae = np.array(
            [[1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4],
             [5, 5, 5, 5, 5, 5]])

        result_mock = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 2.8284271247461903, 4.242640687119285,
              5.656854249492381],
             [2.0, 2.0, 2.0, 2.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              4.242640687119285],
             [3.0, 3.0, 3.0, 3.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              2.8284271247461903],
             [4.0, 4.0, 4.0, 4.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              4.242640687119285],
             [5.0, 5.0, 5.0, 5.0, 1.4142135623730951, 2.8284271247461903, 4.242640687119285,
              5.656854249492381]])

        result_output = utils.enhance_minutiae_points(mock_minutiae)

        np.testing.assert_array_equal(result_output, result_mock)


class PreprocessPredictInputTest(unittest.TestCase):
    def test_correct_input_with_current_setup(self):
        mock_minutiae = np.array(
            [[1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4],
             [5, 5, 5, 5, 5, 5]])

        mock_enhanced_minutiae = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 2.8284271247461903, 4.242640687119285,
              5.656854249492381],
             [2.0, 2.0, 2.0, 2.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              4.242640687119285],
             [3.0, 3.0, 3.0, 3.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              2.8284271247461903],
             [4.0, 4.0, 4.0, 4.0, 1.4142135623730951, 1.4142135623730951, 2.8284271247461903,
              4.242640687119285],
             [5.0, 5.0, 5.0, 5.0, 1.4142135623730951, 2.8284271247461903, 4.242640687119285,
              5.656854249492381]])

        result_mock = [np.array([mock_enhanced_minutiae]), np.array([mock_enhanced_minutiae])]

        result_output = utils.preprocess_predict_input(mock_minutiae, mock_minutiae)

        np.testing.assert_array_equal(result_output, result_mock)
