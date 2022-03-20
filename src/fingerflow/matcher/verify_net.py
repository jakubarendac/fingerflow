from tensorflow.keras import backend

from .VerifyNet import verify_net_model, utils


class VerifyNet:
    def __init__(self, precision, verify_net_path):
        backend.clear_session()

        self.__verify_net = verify_net_model.get_verify_net_model(precision, verify_net_path)

    def verify_fingerprints(self, anchor, sample):
        preprocessed_input = utils.preprocess_predict_input(anchor, sample)

        [[prediction]] = self.__verify_net.predict(preprocessed_input)

        return prediction
