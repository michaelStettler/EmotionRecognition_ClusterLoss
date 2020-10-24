import os
from argparse import ArgumentParser

from PIL import Image


def check_images(path: str):

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        try:
            im = Image.open(file_path)
        except:
            print('corrupt file {}'.format(file))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")

    args = parser.parse_args()
    affectnet_path = args.path

    check_images(affectnet_path)
