import os
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import shutil


def process_affectnet(path: str, new_path: str, file_name: str, new_name: str):

    print('Processing started at {}'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    dataframe = pd.read_csv(path + file_name)

    for index, image_path in enumerate(dataframe.loc[:,
                                       'subDirectory_filePath']):

        directory, image = image_path.split('/')
        dataframe.loc[index, 'subDirectory_filePath'] = image

    dataframe.to_csv(new_path + new_name + "_modified.csv", index=False)

    print('Processing finished at {}'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")
    parser.add_argument("-o", "--old",
                        help="path to the affectnet directory")
    parser.add_argument("-f", "--file",
                        help="file to change")
    parser.add_argument("-n", "--name",
                        help="new name")

    args = parser.parse_args()
    affectnet_path = args.path
    old = args.old
    file = args.file
    name = args.name

    process_affectnet(affectnet_path, old, file, name)
