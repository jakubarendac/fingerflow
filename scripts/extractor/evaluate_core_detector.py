import cv2
# from fingerflow.extractor.core_net import CoreNet
# USING LOCAL VERSION - move script ro parent folder to run it
from src.fingerflow.extractor.core_net import CoreNet

CORE_NET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'
IMG_PATH = '/home/jakub/projects/dp/kernel-detector/test_different_data/2.jpg'
core_net = CoreNet(CORE_NET_PATH)

img = cv2.imread(IMG_PATH)

print("skap detect core => ", core_net.detect_fingerprint_core(img))
