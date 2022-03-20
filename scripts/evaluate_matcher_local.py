import numpy as np
# USING PYPI PACKAGE
# from fingerflow.matcher import Matcher
# USING LOCAL VERSION - move script ro parent folder to run it
from src.fingerflow.matcher import Matcher

WEIGHTS_15 = '/home/jakub/projects/dp/matcher_checkpoints/matcher_contrast_weights_15.h5'
WEIGHTS_20 = '/home/jakub/projects/dp/matcher_checkpoints/matcher_contrast_weights_20.h5'

ANCHOR = '/home/jakub/projects/dp/matcher_training_data/dataset_20_items/04_DB4_A_99_8.csv'
SAMPLE = '/home/jakub/projects/dp/matcher_training_data/dataset_20_items/02_DB1_A_63_7.csv'

matcher = Matcher(20, WEIGHTS_20)

anchor_minutiae = np.genfromtxt(ANCHOR, delimiter=',')
sample_minutiae = np.genfromtxt(SAMPLE, delimiter=',')

matcher.verify(anchor_minutiae, sample_minutiae)
