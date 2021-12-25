from tensorflow.keras import backend

from .CoreNet import core_net_model, utils


class CoreNet:
    def __init__(self, core_net_path):
        backend.clear_session()

        self.__core_net = core_net_model.get_core_net_model(core_net_path)

    def detect_fingerprint_core(self, raw_image_data):
        image_data = utils.preprocess_image_data(raw_image_data[:, :, ::-1])

        prediction_output = self.__core_net.predict(image_data)
        detected_cores = utils.get_detection_data(raw_image_data[:, :, ::-1], prediction_output)

        return detected_cores
