import os
import cv2
import numpy as np
# USING PYPI PACKAGE
# from fingerflow import extractor
# USING LOCAL VERSION - move script ro parent folder to run it
from src.fingerflow import extractor

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"
CORE_NET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'

extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)


def read_image(image_path):
    original_image = np.array(cv2.imread(image_path, 0))
    image_size = np.array(original_image.shape, dtype=np.int32) // 8 * 8
    image = original_image[:image_size[0], :image_size[1]]

    output = dict()

    output['original_image'] = original_image
    output['image_size'] = image_size
    output['image'] = image

    return output


def get_extracted_minutiae_data(image_path):
    image = np.array(cv2.imread(image_path, 0))

    extracted_minutiae = extractor.extract_minutiae(image)

    return extracted_minutiae


def get_extracted_minutiae(image_folder):
    # minutiae_files = []

    for _, _, files in os.walk(image_folder):
        for file_name in files:
            file_path = image_folder + file_name

            _ = get_extracted_minutiae_data(file_path)
            print(f"{file_name} - processed => ")

            # file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]
            # minutiae = Minutiae(file_name_without_extension, minutiae_data)
            # minutiae_files.append(minutiae)

    # return minutiae_files


get_extracted_minutiae(
    '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/test/')
