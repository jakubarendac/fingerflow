import cv2
import numpy as np
# import matplotlib

# from fingerflow.extractor.core_net import CoreNet
# USING LOCAL VERSION - move script ro parent folder to run it
from fingerflow.extractor.core_net import CoreNet

# matplotlib.use('TkAgg')

# '/home/jakub/projects/dp/kernel-detector/trainings/15-12-2021/yolo-kernel_best.weights'
# IOU = 0.2
# {'precision': 0.9810938555030385, 'recall': 0.8827460510328068}
# IOU = 0.5
# {'precision': 0.8730587440918298, 'recall': 0.870121130551817}

# '/home/jakub/projects/dp/kernel-detector/trainings/17-12-2021/yolo-kernel_best.weights'
# IOU = 0.2
# {'precision': 0.9975757575757576, 'recall': 0.98562874251497}
# IOU = 0.5
# {'precision': 0.9303030303030303, 'recall': 0.9846055163566388}

# '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_last.weights'
# IOU = 0.5
# {'precision': 0.7959641255605381, 'recall': 0.9888579387186629}

# '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'
# IOU = 0.2
# {'precision': 0.9992469879518072, 'recall': 0.9836916234247591}
# IOU = 0.5
# {'precision': 0.8576807228915663, 'recall': 0.9810508182601206}

# '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_final.weights'
# IOU = 0.2
# {'precision': 0.9970104633781763, 'recall': 0.9910846953937593}
# IOU = 0.5
# {'precision': 0.7959641255605381, 'recall': 0.9888579387186629}
# IOU = 0.6
# {'precision': 0.5677710843373494, 'recall': 0.9716494845360825}
# IOU = 0.7
# {'precision': 0.2605421686746988, 'recall': 0.9402173913043478}

DATASET_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/datasets/test_local.txt'
WEIGHTS_PATH = '/home/jakub/projects/dp/kernel-detector/trainings/19-12-2021-final/yolo-kernel_best.weights'
IOU_THRESHOLD = 0.7

core_net = CoreNet(WEIGHTS_PATH)


def load_annotations(image_path):
    file_path, _ = image_path.split('.')

    return np.loadtxt(f"{file_path}.txt")


def get_nearest_kernel(detected, ground_truth_x, ground_truth_y):
    nearest_detected_kernel = min(
        detected.iterrows(),
        key=lambda kernel:
        abs(((kernel[1].values[0] + kernel[1].values[2]) / 2) - int(ground_truth_x)) +
        abs(((kernel[1].values[1] + kernel[1].values[3]) / 2) - int(ground_truth_y)))

    return nearest_detected_kernel


def parse_annotation_data(annotation, height, width):
    center_x = annotation[1] * width
    center_y = annotation[2] * height
    annotation_width = annotation[3] * width
    annotation_height = annotation[4] * height

    return {
        'left': center_x - annotation_width / 2,
        'right': center_x + annotation_width / 2,
        'top': center_y - annotation_height / 2,
        'bottom': center_y + annotation_height / 2
    }


def get_iou(ground_truth_kernel, detected_kernel):
    w_intersection = min(ground_truth_kernel['right'], detected_kernel[1].values[2]) - max(
        ground_truth_kernel['left'], detected_kernel[1].values[0])
    h_intersection = min(ground_truth_kernel['bottom'], detected_kernel[1].values[3]) - max(
        ground_truth_kernel['top'], detected_kernel[1].values[1])

    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0

    intersection_area = w_intersection * h_intersection

    ground_truth_area = (ground_truth_kernel['right'] - ground_truth_kernel['left']) * (
        ground_truth_kernel['bottom'] - ground_truth_kernel['top'])

    detected_area = (detected_kernel[1].values[2] - detected_kernel[1].values[0]) * (
        detected_kernel[1].values[3] - detected_kernel[1].values[1])

    union_area = ground_truth_area + detected_area - intersection_area

    return intersection_area / union_area


def get_precision_and_recall(t_p, f_p, f_n):
    if t_p:
        precision = t_p / (t_p + f_p)
        recall = t_p / (t_p + f_n)

        return {
            'precision': precision,
            'recall': recall
        }

    return {
        'precision': 0,
        'recall': 0
    }


with open(DATASET_PATH) as f:
    raw_lines = f.readlines()

    image_path_list = list(map(lambda item: item.strip(), raw_lines))

    t_p = 0
    f_p = 0
    f_n = 0

    for image_path in image_path_list:

        print(f"Processing: {image_path}")

        img = cv2.imread(image_path)

        annotations = load_annotations(image_path)
        predictions = core_net.detect_fingerprint_core(img)

        height, width, _ = img.shape

        if annotations.size > 0:
            annotation_data = parse_annotation_data(annotations, height, width)

            if predictions.size > 0:
                nearest_detected_kernel = get_nearest_kernel(
                    predictions, annotation_data['left'], annotation_data['top'])

                iou = get_iou(annotation_data, nearest_detected_kernel)

                print("skap detected iou => ", iou)

                if iou >= IOU_THRESHOLD:
                    t_p += 1

                else:
                    f_p += 1

            else:
                f_n += 1

    measurements = get_precision_and_recall(t_p, f_p, f_n)
    print(measurements)
