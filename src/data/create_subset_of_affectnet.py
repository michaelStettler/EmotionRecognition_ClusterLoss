"""
this script create n_subset folders containing each n_img from affectnet
The n_subset labels to keep has to be selected in the keep_label folder
"""

import pandas as pd
import shutil, os
import numpy as np
from tqdm import tqdm

path = '../../../../Downloads/AffectNet/'  # computer a
# path = '../../../../media/data/AffectNet/'  # computer b
# path = '../../../AffectNet/'  # computer m

train_file_name = 'training_modified.csv'
# train_file_name = 'training_one_batch.csv'
train_dir = 'training'
# train_dir = 'training_one_batch'

val_file_name = 'validation_modified.csv'
# val_file_name = 'validation_one_batch.csv'
val_dir = 'validation'
# val_dir = 'validation_one_batch'

# Affectnet labels
# 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear
# 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
# keep_label = [0, 1, 2, 4, 6]
keep_label = [0, 1, 2, 3, 4, 5, 6, 7]
# keep_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_subset = np.shape(keep_label)[0]
n_img = 150000
# this extra variables are used to create an unbalanced set for the labels specified
extra = None
# extra = 'unbalanced'
extra_label = [1]
extra_nb = 7500

if extra is None:
    name = 'sub%s_%s' % (n_subset, n_subset * n_img)
else:
    name = 'sub%s_%s_%s' % (n_subset, n_subset * n_img, extra)
print("name", name)


# load the files
df_train = pd.read_csv(path + train_file_name)
df_val = pd.read_csv(path + val_file_name)
# print("df.count", df_train.count())
# print("df.count", df_val.count())

# create the new df for the small dataset
new_train_df = pd.DataFrame()
new_val_df = pd.DataFrame()

# create the new directories
train_directory = path + 'train_' + name
if not os.path.exists(train_directory):
    os.mkdir(train_directory)

val_directory = path + 'val_' + name
if not os.path.exists(val_directory):
    os.mkdir(val_directory)

# loop over the files
i = 0
counters = np.zeros((np.shape(keep_label)[0]))
keep_going = True
while keep_going:
    # dir, img = im_path.split('/')
    label = df_train.loc[i, 'expression']

    # conversion of the label to the keep_label
    mapped_label = np.argwhere(keep_label == label)
    # for the label we want to keep and are not yet filled
    if extra is not None:
        if label in extra_label:
            max_img = n_img + extra_nb
        else:
            max_img = n_img
    else:
        max_img = n_img

    if label in keep_label and counters[mapped_label] < max_img:
        # get the directory path
        # keep the initial labeling for the folder to facilitate the image comparison
        directory = train_directory + '/' + str(label)
        # create the directory in case
        if not os.path.exists(directory):
            os.mkdir(directory)

        counters[mapped_label] += 1
        # get the img name and copy past is to the new folder
        img_name = df_train.loc[i, 'subDirectory_filePath']
        shutil.copyfile(path + train_dir + '/' + img_name, directory + '/' + img_name)

    # increment i
    i += 1

    # if there's an extra just loop over the whole file
    if extra is not None:
        if i >= df_train.shape[0]:
            keep_going = False
    else:
        # stop if i is bigger than the csv file count or if all the labels' counter have reach the number of n_img
        if i >= df_train.shape[0] or (counters >= n_img).all():
            keep_going = False

print("total i", i)
print("train counters", counters)
print()

# validation...  and laziness....
# loop over the files
i = 0
counters = np.zeros((np.shape(keep_label)[0]))
keep_going = True
while keep_going:
    label = df_val.loc[i, 'expression']
    mapped_label = np.argwhere(keep_label == label)
    if label in keep_label and counters[mapped_label] < n_img:
        directory = val_directory + '/' + str(label)
        if not os.path.exists(directory):
            os.mkdir(directory)
        counters[mapped_label] += 1
        img_name = df_val.loc[i, 'subDirectory_filePath']
        shutil.copyfile(path + val_dir + '/' + img_name, directory + '/' + img_name)
    i += 1
    if i >= df_val.shape[0] or (counters >= n_img).all():
        keep_going = False

print("total i", i)
print("val counters", counters)
