"""
this script read the n_rows_train number and split the images into separate folders
"""

import pandas as pd
import shutil, os
from tqdm import tqdm

train = False
# path = '../../../../Downloads/AffectNet/'  # computer a
path = '../../../../media/data_processing/AffectNet/'  # computer b
# path = '../../../AffectNet/'  # computer m

if train:
    file_name = 'training_modified.csv'
    dir = 'training'
else:
    file_name = 'validation_modified.csv'
    dir = 'validation'

name = 'small'
# name = 'one_batch'
# name = '12k'

n_rows_train = 2500
df = pd.read_csv(path + file_name, nrows=n_rows_train)

# create the new df for the small dataset
print("df.count", df.count())
new_small_df = pd.DataFrame(df)

# create the new directory
directory = path + dir + '_' + name
if not os.path.exists(directory):
    os.mkdir(directory)

# for i, line in enumerate(tqdm(df)):
for i, line in enumerate(tqdm(new_small_df.iterrows())):
    # dir, img = im_path.split('/')
    img_name = new_small_df.loc[i, 'subDirectory_filePath']
    shutil.copyfile(path + dir + '/' + img_name, directory + '/' + img_name)

# print(new_df)
if train:
    df.to_csv(path + "training_" + name + ".csv", index=False)
else:
    df.to_csv(path + "validation_" + name + ".csv", index=False)
