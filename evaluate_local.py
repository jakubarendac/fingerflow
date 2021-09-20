# from src.fingerflow import extractor
import os
import cv2
import numpy as np
from src.fingerflow.extractor import minutiae_net

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"

# extractor.Extractor()
minutiae_net = minutiae_net.MinutiaeNet(COARSE_NET_PATH, FINE_NET_PATH)


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
    image = read_image(image_path)

    extracted_minutiae = minutiae_net.extract_minutiae_points(
        image['image'], image['original_image'])

    return extracted_minutiae


def get_extracted_minutiae(image_folder):
    # minutiae_files = []

    for subdir, dirs, files in os.walk(image_folder):
        for file_name in files:
            file_path = image_folder + file_name

            minutiae_data = get_extracted_minutiae_data(file_path)
            print(f"{file_name} - processed")

            # file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]
            # minutiae = Minutiae(file_name_without_extension, minutiae_data)
            # minutiae_files.append(minutiae)

    # return minutiae_files


get_extracted_minutiae(
    '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/DB1_A/')
