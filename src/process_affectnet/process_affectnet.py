"""
This script removes a picture that has zero bytes and where causing issues while
running the initial affectnet database
It also remove the path folder as the flow_from_dataframe cannot handle it
"""
import os
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import shutil


def process_affectnet(path: str, train: bool):

    print('Processing started at {}'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    if train:
        file_name = 'Manually_Annotated_file_lists/training.csv'
    else:
        file_name = 'Manually_Annotated_file_lists/validation.csv'

    dataframe = pd.read_csv(path + file_name)

    # locate the picture that causes problems
    idx = None

    processed_images = 0

    for index, image_path in enumerate(dataframe.loc[:,
                                       'subDirectory_filePath']):

        directory, image = image_path.split('/')
        dataframe.loc[index, 'subDirectory_filePath'] = image

        if '29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg' \
                in image:
            idx = index

        # remove the picture from the dataframe
        if idx is not None:
            try:
                print('deleting row number: {}'.format(idx))
                dataframe = dataframe.drop([idx])
            except KeyError:
                print('Could not delete row {}'.format(idx))

        # remove the sub-folder
        if train:
            new_training_path = path + 'training'
            if not os.path.exists(new_training_path):
                os.mkdir(new_training_path)
            if not os.path.exists(new_training_path + '/' + image_path):
                shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                            new_training_path + '/')
        else:
            new_validation_path = path + 'validation'
            if not os.path.exists(new_validation_path):
                os.mkdir(new_validation_path)
            if not os.path.exists(new_validation_path + '/' + image_path):
                shutil.copy(path + 'Manually_Annotated_Images/' + image_path,
                            new_validation_path + '/')

        processed_images = processed_images + 1

        if processed_images % 1000 == 0:
            print('processed images: {}'.format(processed_images))


    # save the new csv file with no sub folders and this weird pictures
    if train:
        dataframe.to_csv("training_modified.csv", index=False)
    else:
        dataframe.to_csv(path + "validation_modified.csv", index=False)

    print('Processing finished at {}'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))


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
