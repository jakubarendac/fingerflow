import os
import random
from shutil import copyfile
from itertools import groupby

ENCONDINGS_PATH = '/home/jakub/projects/dp/matcher_training_data/08022022232546/'

ANCHOR_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/anchor/'
POSITIVE_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/positive/'
NEGATIVE_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/negative/'


def get_random_negative_item(items, anchor_file):
    random.shuffle(items)

    negative_item = list(filter(lambda file_name: file_name.split(
        '_')[0] != anchor_file.split('_')[0], items))[0]

    return negative_item


pair_number = 0

for _, _, files in os.walk(ENCONDINGS_PATH):
    files_copy = files.copy()

    files.sort()

    for key, group in groupby(files, lambda file_name: file_name.split('_')[0]):
        pairs = zip(*(iter(group),) * 2)

        for pair in pairs:
            anchor, positive = pair

            negative = get_random_negative_item(files_copy, anchor)

            src_anchor_file_path = ENCONDINGS_PATH + anchor
            dst_anchor_file_path = f"{ANCHOR_ENCODINGS}{pair_number}.csv"

            src_positive_file_path = ENCONDINGS_PATH + positive
            dst_positive_file_path = f"{POSITIVE_ENCODINGS}{pair_number}.csv"

            src_negative_file_path = ENCONDINGS_PATH + negative
            dst_negative_file_path = f"{NEGATIVE_ENCODINGS}{pair_number}.csv"

            copyfile(src_anchor_file_path, dst_anchor_file_path)
            copyfile(src_positive_file_path, dst_positive_file_path)
            copyfile(src_negative_file_path, dst_negative_file_path)

            pair_number += 1
