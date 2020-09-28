"""
This sxcrip remove a picture that ha zero bytes and where causing issues while running the initial affectnet database
It also remove the path folder as the flow_from_dataframe cannot handle it
"""

import pandas as pd
import shutil, os
from tqdm import tqdm

train = True
# path = '../../../../Downloads/AffectNet/'
path = '../../../../media/data/AffectNet/'

if train:
    train_file_name = 'training.csv'
else:
    train_file_name = 'validation.csv'

# df = pd.read_csv(path + train_file_name, nrows=50)
df = pd.read_csv(path + train_file_name)
# print(df.loc[:, 'subDirectory_filePath'])
print("df.count", df.count())
idx = None
for i, im_path in enumerate(tqdm(df.loc[:, 'subDirectory_filePath'])):
    dir, img = im_path.split('/')
    df.loc[i, 'subDirectory_filePath'] = img
    # print(img)
    # print(df.ix[i, 'subDirectory_filePath'])
    # print()

    # move images to a single directory
    # if not os.path.exists(dir):
    #     os.makedirs(dir)

    if '29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg' in img:
        # if '29a3' in img:
        #     print(img)
        #     print("FOUND ITTTT")
        #     print(dir)
        idx = i

if idx is not None:
    print("idx", idx)
    df = df.drop([idx])

print("df.count", df.count())
    # if train:
    #     shutil.copy(path+'Manually_Annotated_Images/'+im_path, path+'training/')
    # else:
    #     shutil.copy(path+'Manually_Annotated_Images/'+im_path, path+'validation/')

# save the nwe csv file with no sub folders and this weird pictures
if train:
    df.to_csv("training_modified.csv", index=False)
else:
    df.to_csv("validation_modified.csv", index=False)
