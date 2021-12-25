import cv2
import numpy as np

from . import constants


def preprocess_image_data(raw_image_data):
    image_data = np.array(cv2.cvtColor(raw_image_data, cv2.COLOR_BGR2GRAY))
    image_size = np.array(image_data.shape, dtype=np.int32) // 8 * 8
    image = image_data[:image_size[0], :image_size[1]]

    output = dict()

    output['original_image'] = image_data
    output['image_size'] = image_size
    output['image'] = image

    return output


def get_minutiae_patch_coordinate(patch_center, image_size):
    start_shift = 0
    end_shift = 0

    if patch_center < constants.PATCH_MINU_RADIO:
        start_shift = constants.PATCH_MINU_RADIO - patch_center

    if patch_center + constants.PATCH_MINU_RADIO > image_size:
        end_shift = patch_center + constants.PATCH_MINU_RADIO - image_size

    patch_start = patch_center - constants.PATCH_MINU_RADIO
    patch_end = patch_center + constants.PATCH_MINU_RADIO

    return patch_start + start_shift - end_shift, patch_end + start_shift - end_shift


def get_minutiae_patch(x, y, image):
    x_start, x_end = get_minutiae_patch_coordinate(int(x), image.shape[0])
    y_start, y_end = get_minutiae_patch_coordinate(int(y), image.shape[1])
    patch_minu = image[x_start:x_end, y_start:y_end]

    return patch_minu


def resize_minutiae_patch(minutiae_patch):
    minutiae_patch = cv2.resize(minutiae_patch, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

    ret = np.empty(constants.INPUT_SHAPE, dtype=np.uint8)

    ret[:, :, 0] = minutiae_patch
    ret[:, :, 1] = minutiae_patch
    ret[:, :, 2] = minutiae_patch

    minutiae_patch = ret
    minutiae_patch = np.expand_dims(minutiae_patch, axis=0)

    return minutiae_patch
