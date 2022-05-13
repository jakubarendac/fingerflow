import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.fingerflow.matcher import Matcher


from scripts.matcher_evaluation.utils import utils

matplotlib.use('TkAgg')

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://www.datasciencecentral.com/roc-curve-explained-in-one-picture/
PRECISION = 30
DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/server_dataset/all/{PRECISION}/test/'
WEIGHTS = f'/home/jakub/projects/dp/fingerflow/models/final/weights_{PRECISION}.h5'
ROC_DATA_FOLDER = f'/home/jakub/projects/dp/fingerflow/roc/{PRECISION}'

os.mkdir(ROC_DATA_FOLDER)

matcher = Matcher(PRECISION, WEIGHTS)

pairs, labels = utils.load_dataset(DATASET_PATH)

predictions = matcher.verify_batch(pairs)

fpr, tpr, thresholds = roc_curve(labels, predictions)
auc_value = auc(fpr, tpr)

with open(f'{ROC_DATA_FOLDER}/fpr.npy', 'wb') as f:
    np.save(f, fpr)
    f.close()

with open(f'{ROC_DATA_FOLDER}/tpr.npy', 'wb') as f:
    np.save(f, tpr)
    f.close()

with open(f'{ROC_DATA_FOLDER}/auc_value.npy', 'w') as f:
    f.write(str(auc_value))
    f.close()

plt.plot(fpr, tpr, label=f"VerifyNet - Minutiae: {PRECISION} (AUC: {auc_value})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend(loc=4)
plt.savefig(f'/home/jakub/projects/dp/fingerflow/roc/{PRECISION}.png')
plt.show()
