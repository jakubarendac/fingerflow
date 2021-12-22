import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
# from fingerflow import extractor
from src.fingerflow import extractor

current_datetime = datetime.now().strftime("%d%m%Y%H%M%S")

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"

IMAGES_PATH = '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/test/'
OUTPUT_PATH = f'/home/jakub/projects/dp/matcher_training_data/{current_datetime}/'

extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH)


def load_image_and_extract_minutaie_points(image_path):
    image = np.array(cv2.imread(image_path, 0))

    extracted_minutiae = extractor.extract_minutiae(image)

    return extracted_minutiae


os.mkdir(OUTPUT_PATH)

for _, _, files in os.walk(IMAGES_PATH):
    for file_name in files:
        file_path = IMAGES_PATH + file_name
        extracted_minutiae_points = load_image_and_extract_minutaie_points(file_path)

        output_file_path = f"{OUTPUT_PATH}{Path(file_name).stem}.csv"

        print(output_file_path + " => ")
        with open(output_file_path, "w") as f:
            np.savetxt(f, extracted_minutiae_points, delimiter=',')
