"""
Modify the validation folder of imagenet from number to the synset name to match the training data_processing
"""
import os

folder_path='../../../../Downloads/ImageNet/validation/'

f = open('../utils/map_clsloc.txt')
for line in f:
    words = line.split(' ')
    os.replace(folder_path+words[1], folder_path+words[0])

