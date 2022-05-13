import math
import matplotlib
import matplotlib.pyplot as plt
from src.fingerflow.matcher import Matcher

from scripts.matcher_evaluation.utils import utils

matplotlib.use('TkAgg')

PRECISION = 30
DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/server_dataset/all/{PRECISION}/test/'
WEIGHTS = f'/home/jakub/projects/dp/fingerflow/models/final/weights_{PRECISION}.h5'

matcher = Matcher(PRECISION, WEIGHTS)

positive_pairs, positive_labels = utils.load_dataset(DATASET_PATH, negative=False)
negative_pairs, negative_labels = utils.load_dataset(DATASET_PATH, positive=False)

positive_predictions = matcher.verify_batch(positive_pairs)
negative_predictions = matcher.verify_batch(negative_pairs)

frr = []
far = []
threshold = []

for i in range(100):
    num = 0

    for prediction in positive_predictions:
        if prediction < i/100:
            num += 1

    frr.append(num/len(positive_predictions))
    threshold.append(i)

for i in range(100):
    num = 0

    for prediction in negative_predictions:
        if prediction > i/100:
            num += 1

    far.append(num/len(negative_predictions))

eer_distance = math.inf
eer = math.inf
eer_index = None

for i in range(100):
    cur_frr = frr[i]
    cur_far = far[i]

    cur_eer_distance = abs(cur_frr - cur_far)

    if cur_eer_distance < eer_distance:
        eer_distance = cur_eer_distance
        eer = cur_frr
        eer_index = i

plt.plot(threshold, frr, '--b', label='FRR')
plt.plot(threshold, far, '--g', label='FAR')
plt.plot(eer_index, eer, 'ro', label=f'EER: {round(eer, 3)} (Threshold: {eer_index})')
plt.ylabel('Error rate')
plt.xlabel('Sensitivity')
plt.title('FRR - EER - FRR')
plt.axis([0, 100, 0, 1])
plt.legend(loc=4)
plt.savefig(f'/home/jakub/projects/dp/fingerflow/far_frr/{PRECISION}.png')
plt.show()
