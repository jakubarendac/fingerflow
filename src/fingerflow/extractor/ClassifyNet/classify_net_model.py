from ..MinutiaeNet.FineNet import fine_net_model
from .. import constants


def get_classify_net_model(pretrained_path, num_classes=constants.MINUTIAE_CLASSES):
    return fine_net_model.get_fine_net_model(
        num_classes, pretrained_path, constants.INPUT_SHAPE, "ClassifyNet", "classification_layer")
