import os
import shutil
from argparse import ArgumentParser

import pandas as pd


def rename_csv(path: str, old_name: str, new_name: str):

    # read the csv
    dataframe = pd.read_csv(path + old_name + ".csv")
    # copy the limited amount of images from the old into the new directory
    for index, row in enumerate(dataframe.iterrows()):
        number = dataframe.loc[index, 'expression']
        if number == 0:
            dataframe.loc[index, 'expression'] = 'Neutral'
        if number == 1:
            dataframe.loc[index, 'expression'] = 'Happy'
        if number == 2:
            dataframe.loc[index, 'expression'] = 'Sad'
        if number == 3:
            dataframe.loc[index, 'expression'] = 'Surprise'
        if number == 4:
            dataframe.loc[index, 'expression'] = 'Fear'
        if number == 5:
            dataframe.loc[index, 'expression'] = 'Disgust'
        if number == 6:
            dataframe.loc[index, 'expression'] = 'Anger'
        if number == 7:
            dataframe.loc[index, 'expression'] = 'Contempt'
        if number == 8:
            dataframe.loc[index, 'expression'] = 'None'
        if number == 9:
            dataframe.loc[index, 'expression'] = 'Uncertain'
        if number == 10:
            dataframe.loc[index, 'expression'] = 'Non-Face'

    if index % 1000 == 0:
        print('processed rows: {}'.format(index))

    dataframe.to_csv(path + new_name + ".csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")
    parser.add_argument("-o", "--old",
                        help="old name")
    parser.add_argument("-n", "--new",
                        help="new name")

    args = parser.parse_args()
    affectnet_path = args.path
    old = args.old
    new = args.new

    rename_csv(affectnet_path, old, new)
