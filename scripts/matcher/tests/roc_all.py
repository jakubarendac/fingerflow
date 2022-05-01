import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

VECTORS = [30]
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:cyan', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'darkgreen']
for vector, color in zip(VECTORS, COLORS):
    fpr = np.load(f'/home/jakub/projects/dp/fingerflow/roc/{vector}/fpr.npy')
    tpr = np.load(f'/home/jakub/projects/dp/fingerflow/roc/{vector}/tpr.npy')
    with open(f'/home/jakub/projects/dp/fingerflow/roc/{vector}/auc_value.npy', 'r') as f:
        auc_value = round(float(f.readline().strip()), 3)

        plt.plot(
            fpr,
            tpr,
            color=color,
            label=f"VerifyNet - Minutiae: {vector} (AUC: {auc_value})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend(loc=4)
plt.show()
