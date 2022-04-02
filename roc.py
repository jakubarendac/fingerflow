import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.fingerflow.matcher import Matcher


from scripts.matcher_evaluation.utils import utils

matplotlib.use('TkAgg')

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

DATASET_PATH = '/home/jakub/projects/dp/matcher_training_data/test_20_dataset/'
WEIGHTS = '/home/jakub/projects/dp/fingerflow/models/matcher_contrast_weights_20_20220328-231635.h5'
PRECISION = 20

matcher = Matcher(PRECISION, WEIGHTS)

pairs, labels = utils.load_dataset(DATASET_PATH)

predictions = matcher.verify_batch(pairs)

fpr, tpr, thresholds = roc_curve(labels, predictions)
auc_value = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC: {auc_value} (Minutiae: {PRECISION})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend(loc=4)
plt.show()
