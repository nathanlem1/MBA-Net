"""
Forensic anthropology is best defined as the identification of the human, or what remains of human, for medico-legal
purposes ;)

This code is used for splitting 11k data set to train, val and test split.
train_all is a combination of train and val i.e. it includes val in the training set - just for experiment.

"""

import csv
import os
from shutil import copyfile

import numpy as np

# You only need to change this line to your dataset path
data_path = './11k'
ext = 'jpg'
if not os.path.isdir(data_path):
    print('Please change the 11k_data_path!')

train_path = data_path + '/Hands'   # 11k Hands data set has 190 identities.
if not os.path.isdir(train_path):
    print('Please check if train_path exists or is set correctly!')

# Read Hands info
hand_info = data_path + '/HandInfo.csv'
hand_info_file = open(hand_info, 'r')
reader = csv.DictReader(hand_info_file)

# ------------------- Train-Val-Test split -----------------------------------------------------------------------------
# Train - the first half identities with N-1 samples per identity where N is the number of samples per that identity.
# Val - the first half identities with 1 sample per identity chosen randomly
# Train_all - Train + Val
# Test - the second half identities with all samples per identity

# For Dorsal
save_path_dorsal_r = data_path + '/train_val_test_split_dorsal_r'  # Dorsal right
if not os.path.isdir(save_path_dorsal_r):
    os.mkdir(save_path_dorsal_r)

save_path_dorsal_l = data_path + '/train_val_test_split_dorsal_l'   # Dorsal left
if not os.path.isdir(save_path_dorsal_l):
    os.mkdir(save_path_dorsal_l)

train_all_save_path_dorsal_r = save_path_dorsal_r + '/train_all'
train_save_path_dorsal_r = save_path_dorsal_r + '/train'
val_save_path_dorsal_r = save_path_dorsal_r + '/val'
test_save_path_dorsal_r = save_path_dorsal_r + '/test'
if not os.path.isdir(train_all_save_path_dorsal_r):
    os.mkdir(train_all_save_path_dorsal_r)
if not os.path.isdir(train_save_path_dorsal_r):
    os.mkdir(train_save_path_dorsal_r)
    os.mkdir(val_save_path_dorsal_r)
    os.mkdir(test_save_path_dorsal_r)

train_all_save_path_dorsal_l = save_path_dorsal_l + '/train_all'
train_save_path_dorsal_l = save_path_dorsal_l + '/train'
val_save_path_dorsal_l = save_path_dorsal_l + '/val'
test_save_path_dorsal_l = save_path_dorsal_l + '/test'
if not os.path.isdir(train_all_save_path_dorsal_l):
    os.mkdir(train_all_save_path_dorsal_l)
if not os.path.isdir(train_save_path_dorsal_l):
    os.mkdir(train_save_path_dorsal_l)
    os.mkdir(val_save_path_dorsal_l)
    os.mkdir(test_save_path_dorsal_l)


# For Palmar
save_path_palmar_r = data_path + '/train_val_test_split_palmar_r'  # Palmar right
if not os.path.isdir(save_path_palmar_r):
    os.mkdir(save_path_palmar_r)

save_path_palmar_l = data_path + '/train_val_test_split_palmar_l'  # Palmar left
if not os.path.isdir(save_path_palmar_l):
    os.mkdir(save_path_palmar_l)


train_all_save_path_palmar_r = save_path_palmar_r + '/train_all'
train_save_path_palmar_r = save_path_palmar_r + '/train'
val_save_path_palmar_r = save_path_palmar_r + '/val'
test_save_path_palmar_r = save_path_palmar_r + '/test'
if not os.path.isdir(train_all_save_path_palmar_r):
    os.mkdir(train_all_save_path_palmar_r)
if not os.path.isdir(train_save_path_palmar_r):
    os.mkdir(train_save_path_palmar_r)
    os.mkdir(val_save_path_palmar_r)
    os.mkdir(test_save_path_palmar_r)

train_all_save_path_palmar_l = save_path_palmar_l + '/train_all'
train_save_path_palmar_l = save_path_palmar_l + '/train'
val_save_path_palmar_l = save_path_palmar_l + '/val'
test_save_path_palmar_l = save_path_palmar_l + '/test'
if not os.path.isdir(train_all_save_path_palmar_l):
    os.mkdir(train_all_save_path_palmar_l)
if not os.path.isdir(train_save_path_palmar_l):
    os.mkdir(train_save_path_palmar_l)
    os.mkdir(val_save_path_palmar_l)
    os.mkdir(test_save_path_palmar_l)


print('---------- Data preparation has started ----------------')

# train_all (train + val) and test split. The first half identities for train_all (dorsal right - 72, dorsal left - 73,
# palmar right - 72 and palmar left - 76) and the last identities for test (dorsal right - 71, dorsal left - 73, palmar
# right - 71 and palmar left - 75). By keeping the first half identities for training, we cut and paste the last rest of
# identities into a test folder

count_id = 0
for row in reader:
    id = row['id']
    accessories = row['accessories']
    imageName = row['imageName']
    aspectOfHand = row['aspectOfHand']
    src_path = train_path + '/' + imageName
    if int(accessories) == 0 and aspectOfHand == 'dorsal right':  # Exclude any hand image with accessories
        if int(id) <= 1050:  # train_all
            dst_path_dorsal = train_all_save_path_dorsal_r + '/' + id
        else:  # test
            if int(id) == 1200000:   # To change '1200000 ' to '1200000'. One image has this deviation!
                id = '1200000'
            dst_path_dorsal = test_save_path_dorsal_r + '/' + id
        if not os.path.isdir(dst_path_dorsal):
            os.mkdir(dst_path_dorsal)
        copyfile(src_path, dst_path_dorsal + '/' + imageName)
    elif int(accessories) == 0 and aspectOfHand == 'dorsal left':
        if int(id) <= 1037:
            dst_path_dorsal = train_all_save_path_dorsal_l + '/' + id
        else:
            dst_path_dorsal = test_save_path_dorsal_l + '/' + id
        if not os.path.isdir(dst_path_dorsal):
            os.mkdir(dst_path_dorsal)
        copyfile(src_path, dst_path_dorsal + '/' + imageName)

    elif int(accessories) == 0 and aspectOfHand == 'palmar right':  # Exclude any hand image with accessories
        if int(id) <= 1051:
            dst_path_palmar = train_all_save_path_palmar_r + '/' + id
        else:
            dst_path_palmar = test_save_path_palmar_r + '/' + id
        if not os.path.isdir(dst_path_palmar):
            os.mkdir(dst_path_palmar)
        copyfile(src_path, dst_path_palmar + '/' + imageName)
    elif int(accessories) == 0 and aspectOfHand == 'palmar left':
        if int(id) <= 1042:
            dst_path_palmar = train_all_save_path_palmar_l + '/' + id
        else:
            dst_path_palmar = test_save_path_palmar_l + '/' + id
        if not os.path.isdir(dst_path_palmar):
            os.mkdir(dst_path_palmar)
        copyfile(src_path, dst_path_palmar + '/' + imageName)


# train_val for Dorsal i.e. split train_all (after the test part is removed) into train and val
for root, dirs, files in os.walk(train_all_save_path_dorsal_r, topdown=True):  # Dorsal right
    for dir_name in dirs:
        for root_n, dirs_n, files_n in os.walk(train_all_save_path_dorsal_r + '/' + dir_name):
            val_ind = np.random.randint(0, len(files_n))  # Take randomly
            val_file_name = files_n[val_ind]
            for name in files_n:
                if not name[-3:] == ext:
                    continue
                ID = name.split('_')
                src_path = train_all_save_path_dorsal_r + '/' + dir_name + '/' + name
                dst_path = train_save_path_dorsal_r + '/' + dir_name
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if name == val_file_name:
                    dst_path = val_save_path_dorsal_r + '/' + dir_name  # This image is used as val image
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

for root, dirs, files in os.walk(train_all_save_path_dorsal_l, topdown=True):  # Dorsal left
    for dir_name in dirs:
        for root_n, dirs_n, files_n in os.walk(train_all_save_path_dorsal_l + '/' + dir_name):
            val_ind = np.random.randint(0, len(files_n))  # Take randomly
            val_file_name = files_n[val_ind]
            for name in files_n:
                if not name[-3:] == ext:
                    continue
                ID = name.split('_')
                src_path = train_all_save_path_dorsal_l + '/' + dir_name + '/' + name
                dst_path = train_save_path_dorsal_l + '/' + dir_name
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if name == val_file_name:
                    dst_path = val_save_path_dorsal_l + '/' + dir_name  # This image is used as val image
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)


# train_val for Palmar i.e. split train_all into train and val
for root, dirs, files in os.walk(train_all_save_path_palmar_r, topdown=True):  # Palmar right
    for dir_name in dirs:
        for root_n, dirs_n, files_n in os.walk(train_all_save_path_palmar_r + '/' + dir_name):
            val_ind = np.random.randint(0, len(files_n))  # Take randomly
            val_file_name = files_n[val_ind]
            for name in files_n:
                if not name[-3:] == ext:
                    continue
                ID = name.split('_')
                src_path = train_all_save_path_palmar_r + '/' + dir_name + '/' + name
                dst_path = train_save_path_palmar_r + '/' + dir_name
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if name == val_file_name:
                    dst_path = val_save_path_palmar_r + '/' + dir_name  # This image is used as val image
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

for root, dirs, files in os.walk(train_all_save_path_palmar_l, topdown=True):  # Palmar left
    for dir_name in dirs:
        for root_n, dirs_n, files_n in os.walk(train_all_save_path_palmar_l + '/' + dir_name):
            val_ind = np.random.randint(0, len(files_n))  # Take randomly
            val_file_name = files_n[val_ind]
            for name in files_n:
                if not name[-3:] == ext:
                    continue
                ID = name.split('_')
                src_path = train_all_save_path_palmar_l + '/' + dir_name + '/' + name
                dst_path = train_save_path_palmar_l + '/' + dir_name
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if name == val_file_name:
                    dst_path = val_save_path_palmar_l + '/' + dir_name  # This image is used as val image
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)


# ------ Query and Gallery split from test data ------------------------------------------------------------------------
# A randomly chosen image is put in gallery folder and the rest are put in query folder. This helps to compare each
# query image to each gallery image and then evaluate using Cumulative Matching Characteristics
# (CMC-1 or rank-1 accuracy) i.e. top-1 accuracy (verification accuracy). Here you can generate query_i and gallery_i
# where i = 0, ..., 9 for 10 Monte Carlo runs for proper evaluations.

# Set N = 10 for Monte Carlo runs
N = 10
for i in range(N):
    query_i = 'query' + str(i)
    gallery_i = 'gallery' + str(i)

    # For Dorsal of 11k
    query_save_path_dorsal_r = save_path_dorsal_r + '/' + query_i  # query0, query1, etc.  # Dorsal right
    gallery_save_path_dorsal_r = save_path_dorsal_r + '/' + gallery_i  # gallery0, gallery1, etc
    if not os.path.isdir(gallery_save_path_dorsal_r):
        os.mkdir(gallery_save_path_dorsal_r)
        os.mkdir(query_save_path_dorsal_r)

    query_save_path_dorsal_l = save_path_dorsal_l + '/' + query_i  # Dorsal left
    gallery_save_path_dorsal_l = save_path_dorsal_l + '/' + gallery_i
    if not os.path.isdir(gallery_save_path_dorsal_l):
        os.mkdir(gallery_save_path_dorsal_l)
        os.mkdir(query_save_path_dorsal_l)

    for root, dirs, files in os.walk(test_save_path_dorsal_r, topdown=True):     # Dorsal right
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(test_save_path_dorsal_r + '/' + dir_name):
                gallery_ind = np.random.randint(0, len(files_n))
                gallery_file_name = files_n[gallery_ind]
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    ID = name.split('_')
                    src_path = test_save_path_dorsal_r + '/' + dir_name + '/' + name
                    dst_path = query_save_path_dorsal_r + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    if name == gallery_file_name:
                        dst_path = gallery_save_path_dorsal_r + '/' + dir_name  # This is used as gallery image
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(test_save_path_dorsal_l, topdown=True):   # Dorsal left
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(test_save_path_dorsal_l + '/' + dir_name):
                gallery_ind = np.random.randint(0, len(files_n))
                gallery_file_name = files_n[gallery_ind]
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    ID = name.split('_')
                    src_path = test_save_path_dorsal_l + '/' + dir_name + '/' + name
                    dst_path = query_save_path_dorsal_l + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    if name == gallery_file_name:
                        dst_path = gallery_save_path_dorsal_l + '/' + dir_name  # This is used as gallery image
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    # For Palmar of 11k
    query_save_path_palmar_r = save_path_palmar_r + '/' + query_i  # Palmar right
    gallery_save_path_palmar_r = save_path_palmar_r + '/' + gallery_i
    if not os.path.isdir(gallery_save_path_palmar_r):
        os.mkdir(gallery_save_path_palmar_r)
        os.mkdir(query_save_path_palmar_r)

    for root, dirs, files in os.walk(test_save_path_palmar_r, topdown=True):
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(test_save_path_palmar_r + '/' + dir_name):
                gallery_ind = np.random.randint(0, len(files_n))
                gallery_file_name = files_n[gallery_ind]
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    ID = name.split('_')
                    src_path = test_save_path_palmar_r + '/' + dir_name + '/' + name
                    dst_path = query_save_path_palmar_r + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    if name == gallery_file_name:
                        dst_path = gallery_save_path_palmar_r + '/' + dir_name  # This is used as gallery image
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    query_save_path_palmar_l = save_path_palmar_l + '/' + query_i  # Palmar left
    gallery_save_path_palmar_l = save_path_palmar_l + '/' + gallery_i
    if not os.path.isdir(gallery_save_path_palmar_l):
        os.mkdir(gallery_save_path_palmar_l)
        os.mkdir(query_save_path_palmar_l)
    #
    for root, dirs, files in os.walk(test_save_path_palmar_l, topdown=True):
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(test_save_path_palmar_l + '/' + dir_name):
                gallery_ind = np.random.randint(0, len(files_n))
                gallery_file_name = files_n[gallery_ind]
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    ID = name.split('_')
                    src_path = test_save_path_palmar_l + '/' + dir_name + '/' + name
                    dst_path = query_save_path_palmar_l + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    if name == gallery_file_name:
                        dst_path = gallery_save_path_palmar_l + '/' + dir_name  # This is used as gallery image
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)


# ------ Gallery_all split from galleries produced above ---------------------------------------------------------------
# Prepare gallery_all which is composed of all galleries of dorsal_r, dorsal_l, palmar_r and palmar_l. When preparing
# gallery_all for dorsal_r, add 11,000,000 for dorsal_l, 21,000,000 for palmar_r and 31,000,000 for palmar_l to make the
# IDs unique for evaluation. And then copy gallery_all for each query version per data sets ( dorsal_r, dorsal_l,
# palmar_r and palmar_l).

# Set N = 10 for Monte Carlo runs
N = 10
for i in range(N):
    gallery_i = 'gallery' + str(i)
    gallery_i_all = gallery_i + '_all'

    gallery_save_path_dorsal_r = save_path_dorsal_r + '/' + gallery_i  # gallery1, gallery2, etc
    gallery_save_path_dorsal_l = save_path_dorsal_l + '/' + gallery_i
    gallery_save_path_palmar_r = save_path_palmar_r + '/' + gallery_i
    gallery_save_path_palmar_l = save_path_palmar_l + '/' + gallery_i

    # 1. For right dorsal, gallery1_all, gallery2_all, etc
    gallery_all_path = save_path_dorsal_r + '/' + gallery_i_all  # gallery1_all, gallery2_all, etc
    if not os.path.isdir(gallery_all_path):
        os.mkdir(gallery_all_path)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_r, topdown=True):  # Right dorsal gallery, should be 1st
        # here

        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    src_path = gallery_save_path_dorsal_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_l, topdown=True):  # Left dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 11000000)   # Add 11000000
                    src_path = gallery_save_path_dorsal_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_r, topdown=True):  # Right palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 21000000)   # Add 21000000
                    src_path = gallery_save_path_palmar_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_l, topdown=True):  # Left palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 31000000)   # Add 31000000
                    src_path = gallery_save_path_palmar_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    # 2. For left dorsal, gallery1_all, gallery2_all, etc
    gallery_all_path = save_path_dorsal_l + '/' + gallery_i_all  # gallery1_all, gallery2_all, etc
    if not os.path.isdir(gallery_all_path):
        os.mkdir(gallery_all_path)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_l, topdown=True):  # Left dorsal gallery, should be 1st
        # here

        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    src_path = gallery_save_path_dorsal_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_r, topdown=True):  # Left dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 11000000)  # Add 11000000
                    src_path = gallery_save_path_dorsal_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_r, topdown=True):  # Right palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 21000000)   # Add 21000000
                    src_path = gallery_save_path_palmar_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_l, topdown=True):  # Left palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 31000000)   # Add 31000000
                    src_path = gallery_save_path_palmar_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    # 3. For right palmar, gallery1_all, gallery2_all, etc
    gallery_all_path = save_path_palmar_r + '/' + gallery_i_all  # gallery1_all, gallery2_all, etc
    if not os.path.isdir(gallery_all_path):
        os.mkdir(gallery_all_path)

    for root, dirs, files in os.walk(gallery_save_path_palmar_r, topdown=True):  # Right palmar gallery, should be 1st
        # here

        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    src_path = gallery_save_path_palmar_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_l, topdown=True):  # Left dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 11000000)  # Add 11000000
                    src_path = gallery_save_path_dorsal_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_r, topdown=True):  # Right dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 21000000)  # Add 21000000
                    src_path = gallery_save_path_dorsal_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_l, topdown=True):  # Left palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 31000000)  # Add 31000000
                    src_path = gallery_save_path_palmar_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    # 4. For left palmar, gallery_all, gallery1_all, etc
    gallery_all_path = save_path_palmar_l + '/' + gallery_i_all  # gallery1_all, gallery1_all, etc
    if not os.path.isdir(gallery_all_path):
        os.mkdir(gallery_all_path)

    for root, dirs, files in os.walk(gallery_save_path_palmar_l, topdown=True):  # Left palmar gallery, should be 1st
        # here

        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    src_path = gallery_save_path_palmar_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_palmar_r, topdown=True):  # Right palmar gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_palmar_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 11000000)  # Add 11000000
                    src_path = gallery_save_path_palmar_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_r, topdown=True):  # Right dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_r + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 21000000)  # Add 21000000
                    src_path = gallery_save_path_dorsal_r + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    for root, dirs, files in os.walk(gallery_save_path_dorsal_l, topdown=True):  # Left dorsal gallery
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(gallery_save_path_dorsal_l + '/' + dir_name):
                for name in files_n:
                    if not name[-3:] == ext:
                        continue
                    dir_name_new = str(int(dir_name) + 31000000)  # Add 31000000
                    src_path = gallery_save_path_dorsal_l + '/' + dir_name + '/' + name
                    dst_path = gallery_all_path + '/' + dir_name_new
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)
