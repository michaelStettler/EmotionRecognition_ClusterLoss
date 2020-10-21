"""
This script removes a picture that has zero bytes and where causing issues while
running the initial affectnet database
It also remove the path folder as the flow_from_dataframe cannot handle it
"""
import os
from argparse import ArgumentParser

import pandas as pd
import shutil


def process_affectnet(path: str, train: bool):

    if train:
        file_name = 'training.csv'
    else:
        file_name = 'validation.csv'

    dataframe = pd.read_csv(path + file_name)

    # locate the picture that causes problems
    idx = None
    for index, image_path in enumerate(dataframe.loc[:,
                                       'subDirectory_filePath']):
        print(image_path)
        directory, image = image_path.split('/')
        dataframe.loc[index, 'subDirectory_filePath'] = image

        if '29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg' \
                in image:
            idx = index

        # remove the picture from the dataframe
        if idx is not None:
            dataframe = dataframe.drop([idx])

        # remove the sub-folder
        if train:
            new_training_path = path + 'training'
            if not os.path.exists(new_training_path):
                os.mkdir(new_training_path)
            shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                        new_training_path + '/')
        else:
            new_validation_path = path + 'validation'
            if not os.path.exists(new_validation_path):
                os.mkdir(new_validation_path)
            shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                        new_validation_path + '/')

    # save the new csv file with no sub folders and this weird pictures
    if train:
        dataframe.to_csv("training_modified.csv", index=False)
    else:
        dataframe.to_csv(path + "validation_modified.csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")
    parser.add_argument("-t", "--train", type=bool,
                        help="use to create training dir, else not")

    args = parser.parse_args()
    affectnet_path = args.path
    training_bool = args.train

    process_affectnet(affectnet_path, training_bool)
