"""
This script removes a picture that has zero bytes and where causing issues while
running the initial affectnet database
It also remove the path folder as the flow_from_dataframe cannot handle it
"""

import pandas as pd
import shutil


def process_affectnet(path, train: bool):
    if train:
        file_name = 'training.csv'
    else:
        file_name = 'validation.csv'

    dataframe = pd.read_csv(path + file_name)

    # locate the picture that causes problems
    idx = None
    for index, image_path in enumerate(dataframe.loc[:,
                                       'subDirectory_filePath']):
        directory, image = image_path.split('/')
        dataframe.loc[index, 'subDirectory_filePath'] = image

        if '29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg' \
                in image:
            idx = index

    # remove the picture from the dataframe
    if idx is not None:
        df = dataframe.drop([idx])

    # remove the sub-folder
    if train:
        shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                    path + 'training/')
    else:
        shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                    path + 'validation/')

    # save the new csv file with no sub folders and this weird pictures
    if train:
        df.to_csv("training_modified.csv", index=False)
    else:
        df.to_csv("validation_modified.csv", index=False)
