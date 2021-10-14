import numpy as np
import cv2
from tensorflow.keras import optimizers

from . import constants, utils
from .MinutiaeNet.CoarseNet import coarse_net_model, minutiae_net_utils, coarse_net_utils
from .MinutiaeNet.FineNet import fine_net_model


class MinutiaeNet:
    def __init__(self, coarse_net_path, fine_net_path):
        # Load CoarseNet model
        self.__coarse_net = coarse_net_model.get_coarse_net_model(
            (None, None, 1), coarse_net_path, mode='deploy')

        # Load FineNet model
        self.__fine_net = fine_net_model.get_fine_net_model(num_classes=2,
                                                            pretrained_path=fine_net_path,
                                                            input_shape=constants.INPUT_SHAPE)

        self.__fine_net.compile(loss='categorical_crossentropy',
                                optimizer=optimizers.Adam(learning_rate=0),
                                metrics=['accuracy'])

    # TODO : refactor - too long function
    def extract_minutiae_points(self, image, original_image):  # pylint: disable=too-many-locals
        # Generate OF
        texture_img = minutiae_net_utils.fast_enhance_texture(image, sigma=2.5, show=False)
        dir_map, _ = minutiae_net_utils.get_maps_stft(
            texture_img, patch_size=64, block_size=16, preprocess=True)

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        _, _, _, _, _, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = self.__coarse_net.predict(
            image)

        # Use for output mask
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # ========== Adaptive threshold ==================
        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05

        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = coarse_net_utils.label2mnt(
                np.squeeze(mnt_s_out) *
                np.round(
                    np.squeeze(seg_out)),
                mnt_w_out,
                mnt_h_out,
                mnt_o_out,
                thresh=early_minutiae_thres)

            mnt_nms_1 = minutiae_net_utils.py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = minutiae_net_utils.nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break

            final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
            early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = minutiae_net_utils.fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]

        mnt_refined = []

        # ======= Verify using FineNet ============
        for idx_minu, minutiae in enumerate(mnt_nms):
            # Extract patch from image

            x, y = minutiae[:2]

            patch_minu = utils.get_minutiae_patch(x, y, original_image)

            if patch_minu.size > 0:
                patch_minu = utils.resize_minutiae_patch(patch_minu)

                # Use soft decision: merge FineNet score with CoarseNet score
                [is_minutiae_prob] = self.__fine_net.predict(patch_minu)
                is_minutiae_prob = is_minutiae_prob[0]

                tmp_mnt = mnt_nms[idx_minu, :].copy()
                tmp_mnt[3] = (4*tmp_mnt[3] + is_minutiae_prob) / 5
                mnt_refined.append(tmp_mnt)

            else:
                mnt_refined.append(mnt_nms[idx_minu, :])

        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] >
                              final_minutiae_score_threashold, :]

        coarse_net_model.fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        return mnt_nms
