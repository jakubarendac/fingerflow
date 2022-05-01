import os
import pandas as pd

INPUT_PATH = '/home/jakub/projects/dp/matcher_training_data/server_dataset/01042022233218/'

MINUTIAE_TO_USE = [40]

for min_to_use in MINUTIAE_TO_USE:
    output_path = f'/home/jakub/projects/dp/matcher_training_data/preprocessed_dataset_{min_to_use}/'

    os.mkdir(output_path)

    for _, _, files in os.walk(INPUT_PATH):
        for file_name in files:
            print("processing => ", min_to_use, " => ", file_name)
            minutiae = pd.read_csv(f"{INPUT_PATH}{file_name}", delimiter=',', header=None)
            n_nearest_minutiae = minutiae.nsmallest(
                min_to_use, 5)

            n_nearest_minutiae.to_csv(f"{output_path}{file_name}", header=False, index=False)
