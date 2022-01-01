# pylint: disable=no-self-use
import unittest
import numpy as np

from src.fingerflow.extractor import utils


class PreprocessImageDataTest(unittest.TestCase):
    def test_correct_image_data(self):
        mock_input_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_input_data_size = np.array(mock_input_data.shape, dtype=np.int32) // 8 * 8

        mock_output_data = {
            'original_image': mock_input_data[:, :, -1],
            'image_size': mock_input_data_size[:-1],
            'image': mock_input_data[:mock_input_data_size[0], :mock_input_data_size[0], -1]
        }

        result_output_data = utils.preprocess_image_data(mock_input_data)

        np.testing.assert_array_equal(
            result_output_data.get('original_image'),
            mock_output_data.get('original_image'))

        np.testing.assert_array_equal(
            result_output_data.get('image_size'),
            mock_output_data.get('image_size'))

        np.testing.assert_array_equal(
            result_output_data.get('image'),
            mock_output_data.get('image'))


class GetMinutiaePatchCoordinateTest(unittest.TestCase):
    # patch should be aligned to the patch center
    def test_coordinates_without_shift(self):
        mock_patch_center = 50
        mock_image_size = 100

        mock_output_coordinates = (28, 72)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)

    # space between image start edge and patch center is smaller than PATCH_MINU_RATIO
    # patch should be aligned to the image start
    def test_coordinates_with_start_shift(self):
        mock_patch_center = 21
        mock_image_size = 100

        mock_output_coordinates = (0, 44)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)

    # space between image end edge and patch center is smaller than PATCH_MINU_RATIO
    # patch should be aligned to the end start
    def test_coordinates_with_end_shift(self):
        mock_patch_center = 79
        mock_image_size = 100

        mock_output_coordinates = (56, 100)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)


class GetMinutiaePatchTest(unittest.TestCase):
    def test_patch_without_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 50)

        mock_output_patch = mock_input_image[28:72, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 21)

        mock_output_patch = mock_input_image[28:72, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 79)

        mock_output_patch = mock_input_image[28:72, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 50)

        mock_output_patch = mock_input_image[0:44, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 50)

        mock_output_patch = mock_input_image[56:100, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 21)

        mock_output_patch = mock_input_image[0:44, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 21)

        mock_output_patch = mock_input_image[56:100, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 79)

        mock_output_patch = mock_input_image[0:44, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 79)

        mock_output_patch = mock_input_image[56:100, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)


class ResizeMinutiaePatchtest(unittest.TestCase):
    def test_correct_patch_image_data(self):
        mock_input_data = np.zeros((44, 44))

        mock_output_patch = np.zeros((1, 224, 224, 3))

        result_output_patch = utils.resize_minutiae_patch(mock_input_data)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)
