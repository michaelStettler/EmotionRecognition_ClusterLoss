"""
Count the number of images per category and plot the histogram
"""

import numpy as np
import pandas as pd

np.set_printoptions(precision=2, linewidth=200, suppress=True)


def get_stats_from_csv_file(data_folder, label):
    print("mais coucouuuuuuuuuuu")
    counter = np.zeros((11,1))
    df = pd.read_csv(data_folder)
    for i, line in enumerate(df.iterrows()):
        # dir, img = im_path.split('/')
        cat = df.loc[i, label]
        counter[int(cat)] += 1
    print(counter)


if __name__ == '__main__':

    path = '../../../../Downloads/AffectNet/'  # computer a
    # path = '../../../../media/data/AffectNet/'  # computer b
    # path = '../../../AffectNet/'  # computer m

    data_folder = 'training_small.csv'
    csv = True

    if csv:
        get_stats_from_csv_file(path + data_folder, label='expression')
