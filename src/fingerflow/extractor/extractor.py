from .minutiae_net import MinutiaeNet
from .classify_net import ClassifyNet
from .core_net import CoreNet
from . import utils


class Extractor:
    def __init__(self, coarse_net_path, fine_net_path, classify_net_path, core_net_path):
        self.__extraction_module = MinutiaeNet(coarse_net_path, fine_net_path)
        self.__classification_module = ClassifyNet(classify_net_path)
        self.__core_detection_module = CoreNet(core_net_path)

    def extract_minutiae(self, image_data):
        preprocessed_image = utils.preprocess_image_data(image_data)

        extracted_points = self.__extraction_module.extract_minutiae_points(
            preprocessed_image['image'],
            preprocessed_image['original_image'])

        classified_points = self.__classification_module.classify_minutiae_points(
            preprocessed_image['original_image'],
            extracted_points)

        core_coordinates = self.__core_detection_module.detect_fingerprint_core(image_data)

        return {'core': core_coordinates, 'minutiae': classified_points}
