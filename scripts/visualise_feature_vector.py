import pickle
import cv2
import numpy as np
import pandas as pd
# from fingerflow import extractor

# COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
# FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
# CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"
# CORE_NET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'

IMG_PATH = '/home/jakub/projects/bp/DBs/biometric DBs/FVC_Fingerprint_DB/FVC2004/DB4_A/80_2.tif'
DATA_PATH = 'extractor_output.txt'

MINUTIAE_NUM = 20

# extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)

image = cv2.imread(IMG_PATH)

# extracted_minutiae = extractor.extract_minutiae(image)

# with open(DATA_PATH, 'wb') as f:
#     pickle.dump(extracted_minutiae, f)


def get_correct_core_point(core_data):
    correct_core = core_data[core_data.score == core_data.score.max()]

    x = correct_core[['x1', 'x2']].mean(axis=1)
    y = correct_core[['y1', 'y2']].mean(axis=1)

    return pd.DataFrame({'x': x.values, 'y': y.values})


def get_n_nearest_minutiae(extracted_data, core_point):
    minutiae_data = extracted_data['minutiae']

    if len(core_point) == 0 or len(minutiae_data) < MINUTIAE_NUM:
        return pd.DataFrame()

    minutiae_data['core_distance'] = np.linalg.norm(
        minutiae_data[['x', 'y']].values - core_point.values, axis=1)

    return minutiae_data.sort_values(by=['core_distance']).reset_index(drop=True)


def get_minutiae_color(point_class):
    switcher = {
        0: (255, 255, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
        4: (255, 255, ),
        5: (255, 0, 255)
    }

    return switcher.get(point_class)


def show_core_point(core):
    start_point = (core['x1'].values[0], core['y1'].values[0])
    end_point = (core['x2'].values[0], core['y2'].values[0])
    color = (0, 0, 255)

    cv2.rectangle(image, start_point, end_point, color, 2)


def show_minutiae_points(minutiae, core_point):
    radius = 10
    core = (int(core_point['x']), int(core_point['y']))

    for index, point in minutiae.iterrows():
        center = (int(point['x']), int(point['y']))

        print(center, index)
        cv2.circle(image, center, radius, get_minutiae_color(int(point['class'])), 1)

        if index < MINUTIAE_NUM:
            cv2.line(image, core, center, (0, 0, 255), 1)


with open(DATA_PATH, 'rb') as f:
    extracted_minutiae = pickle.load(f)

    core_point = get_correct_core_point(extracted_minutiae['core'])
    enriched_minutiae = get_n_nearest_minutiae(extracted_minutiae, core_point)

    show_core_point(extracted_minutiae['core'])
    show_minutiae_points(enriched_minutiae, core_point)

    print(enriched_minutiae.to_markdown())
    cv2.imshow('fingerpint', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('visualised_feature_vector_3.png', image)
