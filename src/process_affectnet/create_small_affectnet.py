import os
import shutil
from argparse import ArgumentParser

import pandas as pd


def create_small_affectnet(path: str,
                           number_of_rows: int,
                           train: bool):

    if train:
        file_name = 'training_modified_renamed.csv'
        directory = 'training'
    else:
        file_name = 'validation_modified_renamed.csv'
        directory = 'validation'

    # read the csv and create a new dataframe with a limited number of rows
    dataframe = pd.read_csv(path+file_name, nrows=int(number_of_rows))
    small_dataframe = pd.DataFrame(dataframe)

    # create new directory for the small affectnet
    directory_small = path + directory + '_small' + number_of_rows
    if not os.path.exists(directory_small):
        os.mkdir(directory_small)

    # copy the limited amount of images from the old into the new directory
    for index, row in enumerate(small_dataframe.iterrows()):
        image_name = small_dataframe.loc[index, 'subDirectory_filePath']
        shutil.copyfile(path + directory + '/' + image_name,
                        directory_small + '/' + image_name)

    # create a new csv file for the limited amount of images
    if train:
        dataframe.to_csv(path + "training_small" + number_of_rows +
                         ".csv", index=False)
    else:
        dataframe.to_csv(path + "validation_small" + number_of_rows +
                         ".csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")
    parser.add_argument("-n", "--number",
                        help="number of images")
    parser.add_argument("-t", "--train", type=bool,
                        help="include when creating a training directory")

    args = parser.parse_args()
    affectnet_path = args.path
    subset_size = args.number
    training_bool = args.train

    create_small_affectnet(affectnet_path, subset_size, training_bool)
