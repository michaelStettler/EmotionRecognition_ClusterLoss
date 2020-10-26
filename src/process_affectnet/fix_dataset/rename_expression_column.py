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
        elif number == 1:
            dataframe.loc[index, 'expression'] = 'Happy'
        elif number == 2:
            dataframe.loc[index, 'expression'] = 'Sad'
        elif number == 3:
            dataframe.loc[index, 'expression'] = 'Surprise'
        elif number == 4:
            dataframe.loc[index, 'expression'] = 'Fear'
        elif number == 5:
            dataframe.loc[index, 'expression'] = 'Disgust'
        elif number == 6:
            dataframe.loc[index, 'expression'] = 'Anger'
        elif number == 7:
            dataframe.loc[index, 'expression'] = 'Contempt'
        elif number == 8:
            dataframe.loc[index, 'expression'] = 'None'
        elif number == 9:
            dataframe.loc[index, 'expression'] = 'Uncertain'
        elif number == 10:
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
