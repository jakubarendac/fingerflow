
import os
import time
import numpy as np
from datetime import datetime
import cv2
from fingerflow.extractor import Extractor

IMAGE_FOLDER = "/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/DB1_A/"
# IMAGE_FOLDER = "/home/jakubarendac/dataset/FVC2002/Db1_a/"

COARSE_NET_PATH = "/home/jakub/projects/dp/fingerflow/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/dp/fingerflow/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/dp/fingerflow/models/ClassifyNet.h5"
CORE_NET_PATH = "/home/jakub/projects/dp/fingerflow/models/yolo-kernel_final.weights"

date_time = datetime.now().strftime('%Y%m%d-%H%M%S')

output_file_name = 'network_speed_performance_test_' + date_time + '.txt'

total_processing_time = 0
processed_images_count = 0
processed_minutiae_count = 0

output_file = open(output_file_name, 'a')

extractor = Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)

for _, _, files in os.walk(IMAGE_FOLDER):
    for file_name in files:
        file_path = IMAGE_FOLDER + file_name

        image = cv2.imread(file_path)

        print("processing => ", file_name)

        t = time.process_time()
        extracted_data = extractor.extract_minutiae(image)
        elapsed_time = time.process_time() - t

        extracted_minutiae = extracted_data['minutiae']

        output_file.write('File: ' + file_name + '\n')

        if len(extracted_minutiae) > 0:
            total_processing_time += elapsed_time
            processed_minutiae_count += extracted_minutiae.shape[0]
            processed_images_count += 1

            output_file.write('\tprocessing_time: ' + str(elapsed_time) + ' sec.\n')
            output_file.write(
                '\tprocessed_minutiae_count: ' + str(extracted_minutiae.shape[0]) + '\n\n')

        else:
            print('error occured during extraction')
            output_file.write('\terror occured during extraction - no minutiae extracted\n')

average_processing_time_minutiae = total_processing_time / processed_minutiae_count if processed_minutiae_count > 0 else 0

average_processing_time_file = total_processing_time / processed_images_count if processed_images_count > 0 else 0

output_file.write('Total extracted images count: ' + str(processed_images_count) + '\n')
output_file.write('Total extracted minutiae: ' + str(processed_minutiae_count) + '\n')
output_file.write('Total extraction time: ' + str(total_processing_time) + ' sec.\n')
output_file.write(
    'Average extraction time for 1 file: ' + str(average_processing_time_file) + ' sec.\n')
output_file.write('Average extraction time for 1 minutiae: ' +
                  str(average_processing_time_minutiae) + ' sec.\n')

output_file.close()
