#!/usr/bin/env python
# coding: utf-8

"""
Directory Structure
dataset/
    |-- crop_image/
        |-- train/
            |-- *.jpg
            |-- train.txt
            |-- label.txt
        |-- test/
            |-- *.jpg
            |-- train.txt
            |-- label.txt
"""

import shutil
from os import listdir, makedirs
from os.path import join, exists

image_path = './dataset/crop_image/'
name_file_path = './dataset/train.txt'
label_file_path = './dataset/label.txt'
train_path = image_path + 'train/'
test_path = image_path + 'test/'

# for splitting training set and test set
TRAINING_NUM = 1820

if __name__ == '__main__':

    if not exists(train_path):
        makedirs(train_path)
    if not exists(test_path):
        makedirs(test_path)

    image_list = []
    label_list = []

    # from name file, read image names
    f = open(name_file_path, 'r')
    contents = f.readlines()
    for line in contents:
        path = line.strip()
        image_list.append(path)
    f.close()

    # from label file, read labels
    f = open(label_file_path, 'r')
    contents = f.readlines()
    for line in contents:
        path = line.strip()
        label_list.append(path)
    f.close()

    # split training and test set
    train_set = image_list[:TRAINING_NUM]
    test_set = image_list[TRAINING_NUM:]

    # copy images to certain directory
    for name in train_set:
        shutil.copy((image_path + name), (train_path + name))

    for name in test_set:
        shutil.copy((image_path + name), (test_path + name))

    # write name files
    with open(train_path + 'train.txt', 'w') as f:
        for name in train_set:
            f.write("%s\n" % name)

    with open(test_path + 'test.txt', 'w') as f:
        for name in test_set:
            f.write("%s\n" % name)

    train_set = label_list[:TRAINING_NUM]
    test_set = label_list[TRAINING_NUM:]

    # write label files
    with open(train_path + 'label.txt', 'w') as f:
        for name in train_set:
            f.write("%s\n" % name)

    with open(test_path + 'label.txt', 'w') as f:
        for name in test_set:
            f.write("%s\n" % name)

