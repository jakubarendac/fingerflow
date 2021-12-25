from tensorflow import config, distribute
from tensorflow.keras import layers, models

from . import constants, custom_layers, utils


def get_core_net_model(pretrained_path):
    has_multiple_gpu = len(config.list_logical_devices('GPU')) > 1

    if has_multiple_gpu and constants.USE_DISTRIBUTED_TRAINING:
        mirrored_strategy = distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            return build_model(pretrained_path)
    else:
        return build_model(pretrained_path)


def build_model(pretrained_path):
    # core yolo model
    input_layer = layers.Input(constants.INPUT_SHAPE)
    yolov4_output = custom_layers.yolov4_neck(input_layer)
    yolo_core_model = models.Model(input_layer, yolov4_output)

    # Build inference model
    yolov4_output = custom_layers.yolov4_head(yolov4_output)
    # output: [boxes, scores, classes, valid_detections]
    yolo_model = models.Model(input_layer, custom_layers.nms(yolov4_output))

    utils.load_darknet_weights(yolo_core_model, pretrained_path)

    print(f'Core net weights loaded from {pretrained_path}')

    return yolo_model
