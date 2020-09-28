import csv
import os
import time
import math
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import glob

import multiprocessing
from multiprocessing import Pool


def split(filehandler, delimiter=',', row_limit=1000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)


def process_file(args):
    data = []
    labels = []

    with open(args[0], 'r') as csvfile:
        labels_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        row1 = next(labels_reader)
        print(row1)
        row2 = next(labels_reader)
        print(row2)
        print(row2[6])
        labels.append(row2[6])
        img = scipy.ndimage.imread(path + row1[0])
        print(np.shape(img))

        # for row in tqdm(labels_reader):
        #     try:
        #         # todo set the final output
        #         img = scipy.ndimage.imread(args[1] + row[0])
        #         img = scipy.misc.imresize(img, [args[3], args[4]])
        #         data_processing.append(img)
        #
        #         if args[2]:  # extended
        #             labels.append(row[1:5] + row[5].split(';') + row[6:])
        #         else:  # categorical
        #             labels.append(row[6])
        #
        #     except FileNotFoundError:
        #         print("file %s not found" % row[0])

    return data, labels


def load_dataset(file, path, img_height, img_width, extended=False, keep_headers=True):
    data = []
    labels = []

    # first split the csv file for each multi threads
    split(open(file, 'r'),
          output_name_template='tmp_output_%s.csv',
          row_limit=200,
          keep_headers=keep_headers)

    # list all the temp output files
    # print(os.listdir())
    files = glob.glob('tmp_output*.csv')

    # construct a single args tuple for the multi thread process
    args = []
    for file in files:
        args.append([file, path, extended, img_height, img_width])

    # create the multi threading process and run
    p = Pool(multiprocessing.cpu_count())
    results = p.map(process_file, args)
    p.close()
    p.join()

    # merge the results
    for i in tqdm(range(np.shape(results)[0])):
        data += results[i][0]
        labels += results[i][1]

    # delete temp files
    for file in files:
        os.remove(file)

    return np.array(data), np.array(labels)


if __name__ == '__main__':
    # record starting time
    start = time.time()

    path = '../../../../Downloads/AffectNet/'
    train_file_name = 'training.csv'
    # train_file_name = 'training_small.csv'
    # val_file_name = 'validation_small.csv'    # no headers!!!!!
    # val_file_name = 'validation.csv'          # no headers!!!!!
    # print(os.listdir(path))
    data, labels = process_file([path + train_file_name, path + 'Manually_Annotated_Images/'])

    # # first split the csv file for each multi threads
    # split(open(path + train_file_name, 'r'), row_limit=20000, keep_headers=True)
    # print('Done main split')
    #
    # # list all the output files
    # # print(os.listdir())
    # files = glob.glob('output*.csv')
    #
    # for file in files:
    #     # training
    #     data_processing, labels = load_dataset(file, path + 'Manually_Annotated_Images/', 224, 224)
    #     # # data_processing, labels = load_extended_dataset(file, path + 'Manually_Annotated_Images/')
    #     print("print data_processing shape", np.shape(data_processing))
    #     print("print labels shape", np.shape(labels))
    #
    # # validation
    # # data_processing, labels = process_file([path + val_file_name, path + 'Manually_Annotated_Images/', False])
    # # data_processing, labels = load_categorical_dataset(path + val_file_name, path + 'Manually_Annotated_Images/', keep_headers=False)
    #
    #
    # # list all the output files
    # # print(os.listdir())
    # files = glob.glob('output*.csv')
    # # delete temp files
    # for file in files:
    #     os.remove(file)

    total_time = time.time() - start
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print("Total time: %.0f hours %.0f min %.0f sec" % (h, m, s))
