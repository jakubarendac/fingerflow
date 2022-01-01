# from src.fingerflow import extractor
import cv2
import numpy as np

from src.fingerflow import extractor

COARSE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5"
FINE_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5"
CLASSIFY_NET_PATH = "/home/jakub/projects/bp/minutiae_classificator/models/ClassifyNet.h5"
CORE_NET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'

IMG_PATH = '/home/jakub/Desktop/sad dog.jpeg'

extractor = extractor.Extractor(COARSE_NET_PATH, FINE_NET_PATH, CLASSIFY_NET_PATH, CORE_NET_PATH)

image = cv2.imread(IMG_PATH)


print(extractor.extract_minutiae(image))
