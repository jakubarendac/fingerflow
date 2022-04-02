import tensorflow as tf
import numpy as np
from tensorflow.keras import backend

from .VerifyNet import verify_net_model, utils


class VerifyNet:
    def __init__(self, precision, verify_net_path):
        backend.clear_session()

        self.__verify_net = verify_net_model.get_verify_net_model(precision, verify_net_path)

    def verify_fingerprints(self, anchor, sample):
        preprocessed_input = utils.preprocess_predict_input(anchor, sample)

        # change according to docs -> https://keras.io/api/models/model_training_apis/#predict-method
        [[prediction]] = self.__verify_net.predict(preprocessed_input)

        return prediction

    def verify_fingerprints_batch(self, pairs):
        preprocessed_pairs = [utils.preprocess_predict_input(
            anchor, sample) for [anchor, sample] in pairs]

        # print(np.stack(([preprocessed_pairs[0]],
        #                 [preprocessed_pairs[1]]))[0])

        def predict_item(pair):
            [[prediction]] = self.__verify_net.predict(pair)

            return prediction

        predictions = [predict_item(pair) for pair in preprocessed_pairs]

        return predictions
        # TODO : find out how this can work :D
        # return self.__verify_net.predict(
        #     np.vstack((preprocessed_pairs[0],
        #               preprocessed_pairs[1])))
