import os
from datetime import datetime
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from fingerflow import extractor

current_datetime = datetime.now().strftime("%d%m%Y%H%M%S")

MINUTIAE_NUM = 20

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"
CORE_NET_PATH = "/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights"

OUTPUT_PATH = f'/home/jakub/projects/dp/matcher_training_data/{current_datetime}/'

IMAGES_FOLDERS = [
    {
        'name': '04_DB4_A',
        'path': '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/DB4_A/'
    },
    {
        'name': '04_DB3_A',
        'path': '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/DB3_A/'
    },
    {
        'name': '02_DB2_A',
        'path': '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2002/Db2_a/'
    },
    {
        'name': '02_DB1_A',
        'path': '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2002/Db1_a/'
    }
]

extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)


def load_image_and_extract_minutaie_points(image_path):
    image = cv2.imread(image_path)

    extracted_data = extractor.extract_minutiae(image)
    nearest_minutiae = get_n_nearest_minutiae(extracted_data)

    return nearest_minutiae


def get_correct_core_point(core_data):
    correct_core = core_data[core_data.score == core_data.score.max()]

    x = correct_core[['x1', 'x2']].mean(axis=1)
    y = correct_core[['y1', 'y2']].mean(axis=1)

    return pd.DataFrame({'x': x.values, 'y': y.values})


def get_n_nearest_minutiae(extracted_data):
    minutiae_data = extracted_data['minutiae']
    core_point = get_correct_core_point(extracted_data['core'])

    if len(core_point) == 0 or len(minutiae_data) < MINUTIAE_NUM:
        return pd.DataFrame()

    minutiae_data['core_distance'] = np.linalg.norm(
        minutiae_data[['x', 'y']].values - core_point.values, axis=1)

    n_nearest_minutiae = minutiae_data.nsmallest(
        MINUTIAE_NUM, 'core_distance')

    return n_nearest_minutiae


os.mkdir(OUTPUT_PATH)

for image_folder in IMAGES_FOLDERS:
    for _, _, files in os.walk(image_folder['path']):
        for file_name in files:
            file_path = image_folder['path'] + file_name
            extracted_minutiae_points = load_image_and_extract_minutaie_points(file_path)

            output_file_path = f"{OUTPUT_PATH}{image_folder['name']}_{Path(file_name).stem}.csv"

            if extracted_minutiae_points.empty:
                print(
                    f"Not detected core point or not enough extracted minutiae -> file {file_name} is not being used")

            else:
                print("writing file => " + output_file_path)
                extracted_minutiae_points.to_csv(output_file_path, index=False, header=False)
