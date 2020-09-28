"""
Split the affecnet database by category to use it later with flow_from_directory
"""

import pandas as pd
import shutil, os
from tqdm import tqdm

path = '../../../../Downloads/AffectNet/'  # computer a
# path = '../../../../media/data/AffectNet/'  # computer b
# path = '../../../AffectNet/'  # computer m

# name = 'small'
name = 'one_batch'

img_csv_name = 'subDirectory_filePath'
label_csv_name = 'expression'

train_name = path + 'training_'+name+'.csv'
train_dir = path + 'training_'+name
validation_name = path + 'validation_'+name+'.csv'
val_dir = path + 'validation_'+name

# load the csv file
train_df = pd.read_csv(train_name)
val_df = pd.read_csv(validation_name)

# print("train_df.count", train_df.count())
# print("validation_df.count", val_df.count())

# loop over train
for i, line in enumerate(tqdm(train_df.iterrows())):
    # get image name
    img_name = train_df.loc[i, img_csv_name]
    # get image label
    label = train_df.loc[i, label_csv_name]

    # create folder if it does not exist yet
    directory = path + 'categorical_training_one_batch/' + str(label)
    if not os.path.exists(directory):
        os.mkdir(directory)

    shutil.copyfile(train_dir + '/' + img_name, directory + '/' + img_name)

# loop over validation
for i, line in enumerate(tqdm(val_df.iterrows())):
    # get image name
    img_name = val_df.loc[i, img_csv_name]
    # get image label
    label = val_df.loc[i, label_csv_name]

    # create folder if it does not exist yet
    directory = path + 'categorical_validation_one_batch/' + str(label)
    if not os.path.exists(directory):
        os.mkdir(directory)

    shutil.copyfile(val_dir + '/' + img_name, directory + '/' + img_name)
