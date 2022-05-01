import numpy as np
# USING PYPI PACKAGE
# from fingerflow.matcher import Matcher
# USING LOCAL VERSION - move script ro parent folder to run it
from src.fingerflow.matcher import Matcher

WEIGHTS_15 = '/home/jakub/projects/dp/matcher_checkpoints/matcher_contrast_weights_15.h5'
WEIGHTS_20 = '/home/jakub/projects/dp/fingerflow/models/final/weights_20.h5'

ANCHOR = '/home/jakub/projects/dp/fingerflow/src/fingerflow/matcher/test_data/372_11.csv'
SAMPLE = '/home/jakub/projects/dp/fingerflow/src/fingerflow/matcher/test_data/372_12.csv'

matcher = Matcher(20, WEIGHTS_20)
# matcher.plot_model()
anchor_minutiae = np.genfromtxt(ANCHOR, delimiter=',')
sample_minutiae = np.genfromtxt(SAMPLE, delimiter=',')

print(matcher.verify(anchor_minutiae, sample_minutiae))
