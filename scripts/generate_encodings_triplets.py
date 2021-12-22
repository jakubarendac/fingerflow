import os
import random
from shutil import copyfile
from pathlib import Path

ENCONDINGS_PATH = '/home/jakub/projects/dp/matcher_training_data/03112021194631/'

ANCHOR_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/anchor/'
POSITIVE_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/positive/'
NEGATIVE_ENCODINGS = '/home/jakub/projects/dp/matcher_training_data/dataset/negative/'


def should_add_anchor(encodings_item):
    return not os.path.exists(f"{ANCHOR_ENCODINGS}{encodings_item}.csv")


def should_add_positive(encodings_item):
    return not os.path.exists(f"{POSITIVE_ENCODINGS}{encodings_item}.csv")


def shuffle_negative_encodings(encodings_names):
    anchor_files = os.listdir(ANCHOR_ENCODINGS)

    def get_item(anchor_file):
        return Path(anchor_file).stem

    anchor_items = list(map(get_item, anchor_files))
    anchor_items_copy = anchor_items.copy()

    for encoding_name in encodings_names:
        if len(anchor_items_copy) == 0:
            return

        encoding_item = encoding_name.split('_')[0]
        if encoding_item in anchor_items:
            random_item = random.choice(anchor_items)

            while random_item is encoding_item or str(random_item) not in anchor_items_copy:
                random_item = random.choice(anchor_items)

            src_encoding_path = ENCONDINGS_PATH + encoding_name
            dst_encoding_path = f"{NEGATIVE_ENCODINGS}{random_item}.csv"

            # print("chosen => ", random_item, encoding_item)

            copyfile(src_encoding_path, dst_encoding_path)
            anchor_items_copy.remove(random_item)


for _, _, files in os.walk(ENCONDINGS_PATH):
    file_names = files.copy()

    for file_name in files:
        src_file_path = ENCONDINGS_PATH + file_name
        item = file_name.split('_')[0]

        if should_add_anchor(item):
            dst_file_path = f"{ANCHOR_ENCODINGS}{item}.csv"
            copyfile(src_file_path, dst_file_path)
            file_names.remove(file_name)

            continue

        if should_add_positive(item):
            dst_file_path = f"{POSITIVE_ENCODINGS}{item}.csv"
            copyfile(src_file_path, dst_file_path)
            file_names.remove(file_name)

            continue

shuffle_negative_encodings(file_names)
