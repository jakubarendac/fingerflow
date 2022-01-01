import numpy as np
from tensorflow.keras import optimizers

from .ClassifyNet import classify_net_model, utils as classify_net_utils
from . import utils


class ClassifyNet:
    def __init__(self, classify_net_path):
        # Load ClassifyNet model
        self.__classify_net = classify_net_model.get_classify_net_model(classify_net_path)

        self.__classify_net.compile(loss='categorical_crossentropy',
                                    optimizer=optimizers.Adam(learning_rate=0),
                                    metrics=['accuracy'])

    def classify_minutiae_patch(self, minutiae_patch):
        resized_minutiae_patch = utils.resize_minutiae_patch(minutiae_patch)

        [minutiae_classes] = self.__classify_net.predict(resized_minutiae_patch)

        numpy_minutiae_classes = np.array(minutiae_classes)
        minutiae_type = float(np.argmax(numpy_minutiae_classes))

        return minutiae_type

    def classify_minutiae_points(self, image, extracted_minutiae):
        classified_minutiae = list()

        if extracted_minutiae.size != 0:
            for minutiae in extracted_minutiae:
                x, y = minutiae[:2]

                patch_minu = utils.get_minutiae_patch(x, y, image)

                minutiae_type = self.classify_minutiae_patch(patch_minu)

                tmp_mnt = minutiae.copy()
                tmp_mnt[4] = minutiae_type

                classified_minutiae.append(tmp_mnt)

        formatted_data = classify_net_utils.format_classified_data(np.array(classified_minutiae))

        return formatted_data
