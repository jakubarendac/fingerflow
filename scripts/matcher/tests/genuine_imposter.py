import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from src.fingerflow.matcher import Matcher

from scripts.matcher_evaluation.utils import utils

matplotlib.use('TkAgg')

PRECISION = 10
DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/server_dataset/all/{PRECISION}/test/'
WEIGHTS = f'/home/jakub/projects/dp/fingerflow/models/final/weights_{PRECISION}.h5'

matcher = Matcher(PRECISION, WEIGHTS)

positive_pairs, positive_labels = utils.load_dataset(DATASET_PATH, negative=False)
negative_pairs, negative_labels = utils.load_dataset(DATASET_PATH, positive=False)

positive_predictions = matcher.verify_batch(positive_pairs)
negative_predictions = matcher.verify_batch(negative_pairs)

_, ax = plt.subplots()

sns.kdeplot(data=positive_predictions, ax=ax, color='g', label="Genuine distribution")
sns.kdeplot(data=negative_predictions, ax=ax, color='r', label="Imposter distribution")
plt.xlabel('Matching score')
plt.ylabel('Density')
plt.title('Genuine and imposter score distributions')
# plt.axis([0, 1, 0, 1])
plt.legend(loc=2)
plt.savefig(f'/home/jakub/projects/dp/fingerflow/genuine_imposter/{PRECISION}.png')
plt.show()
