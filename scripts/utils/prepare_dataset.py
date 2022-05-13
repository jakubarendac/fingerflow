import os
import shutil

ENCONDINGS_PATH = '/home/jakub/projects/dp/matcher_training_data/23032022235454/'
OUTPUT_PATH = '/home/jakub/projects/dp/matcher_training_data/preprocessed_dataset/'

item_n = 0
item_variation = 0
cur_item = None

for _, _, files in os.walk(ENCONDINGS_PATH):
    files.sort()

    for file_name in files:
        splitted = file_name.split('_')
        item = '_'.join(splitted[:4])

        if not cur_item:
            cur_item = item

        if cur_item == item:
            item_variation += 1

        if cur_item != item:
            cur_item = item
            item_n += 1
            item_variation = 0

        src = f"{ENCONDINGS_PATH}{file_name}"
        dst = f"{OUTPUT_PATH}{item_n}_{item_variation}.csv"

        print(src, dst)
        shutil.copyfile(src, dst)
