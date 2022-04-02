import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from src.fingerflow.matcher import Matcher

from scripts.matcher_evaluation.utils import utils

matplotlib.use('TkAgg')

DATASET_PATH = '/home/jakub/projects/dp/matcher_training_data/test_20_dataset/'
WEIGHTS = '/home/jakub/projects/dp/fingerflow/models/matcher_contrast_weights_20_20220328-231635.h5'
PRECISION = 20

matcher = Matcher(PRECISION, WEIGHTS)

positive_pairs, positive_labels = utils.load_dataset(DATASET_PATH, negative=False)
negative_pairs, negative_labels = utils.load_dataset(DATASET_PATH, positive=False)

positive_predictions = matcher.verify_batch(positive_pairs)
negative_predictions = matcher.verify_batch(negative_pairs)

_, ax = plt.subplots()

sns.kdeplot(data=positive_predictions, ax=ax, color='g')
sns.kdeplot(data=negative_predictions, ax=ax, color='r')
plt.xlabel('Matching score')
plt.ylabel('Frequency')
plt.title('Genuine/Impostor distribution')
plt.axis([0, 1, 0, 1])
# plt.legend(loc=4)
plt.show()
