from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from fingerflow import extractor
from src.fingerflow import extractor

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"
CORE_NET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'

IMG_PATH = '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/sample/11_7.tif'
RADIUS = 5


def get_minutiae_color(point_class):
    switcher = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
        4: (255, 255, ),
        5: (255, 0, 255)
    }

    return switcher.get(point_class)


current_datetime = datetime.now().strftime("%d%m%Y%H%M%S")

extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)

image = cv2.imread(IMG_PATH)

extracted_data = extractor.extract_minutiae(image)

extracted_minutiae = extracted_data['minutiae']
core = extracted_data['core']

start_point = (core['x1'].values[0], core['y1'].values[0])
end_point = (core['x2'].values[0], core['y2'].values[0])
color = (0, 0, 255)

cv2.rectangle(image, start_point, end_point, color, 2)


if len(extracted_minutiae) > 0:
    for index, minutia in extracted_minutiae.iterrows():
        print(minutia)
        x = minutia['x']
        y = minutia['y']
        o = minutia['angle']
        minutia_class = minutia['class']

        start_point = (int(x-RADIUS), int(y-RADIUS))
        end_point = (int(x+RADIUS), int(y+RADIUS))

        color = get_minutiae_color(int(minutia_class))

        cv2.rectangle(image, start_point, end_point, color, 2)

        line_start_point = (int(x), int(y))
        line_end_point = (int(x + 20 * np.cos(o)), int(y + 20 * np.sin(o)))

        cv2.line(image, line_start_point, line_end_point, color, 2)

cv2.imshow('Extractor single image test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'extractor_single_img_test_{current_datetime}.png', image)
