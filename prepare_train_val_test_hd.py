"""
Forensic anthropology is best defined as the identification of the human, or what remains of human, for medico-legal
purposes ;)

This code is used for splitting HD or PolyUKnuckleV1 data set to train, val and test split.
train_all is a combination of train and val i.e. it includes val in the training set - just for experiment.

"""

import os
from shutil import copyfile

import numpy as np


# Set to your dataset path
data_path = './HD/Original Images'
ext = 'jpg'
if not os.path.isdir(data_path):
    print('Please change the HUnique_data_path!')


train_path = data_path + '/1-501'  # HD 1-501 has 501 identities.
add2gallery_path = data_path + '/502-712'  # Images of 211 subjects to add to each gallery!
if not os.path.isdir(train_path):
    print('Please check if train_path exists or is set correctly!')


# ------------------- Train-Val-Test split -----------------------------------------------------------------------------
# For the PolyUKnuckleV1 data set:
# Train - the first 400 identities with 4 samples per identity
# Val - the first 400 identities with 1 sample per identity chosen randomly
# Train_all - Train + Val
# Test - the last 103 identities with all 5 samples per identity

# For the HD data set:
# Train - the first 251 identities.
# Val - the first 251 identities with 1 sample per identity chosen randomly
# Train_all - Train + Val
# Test - the last 251 identities with all samples per identity

save_path = data_path + '/train_val_test_split'  # Set path here!
if not os.path.isdir(save_path):
    os.mkdir(save_path)

train_all_save_path = save_path + '/train_all'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
test_save_path = save_path + '/test'
if not os.path.isdir(train_all_save_path):
    os.mkdir(train_all_save_path)
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)
    os.mkdir(test_save_path)


print('---------- Data preparation has started ----------------')

# train_all (train + val) and test split. The first 251 identities for train_all and the last 251 identities for test
for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:].lower() == ext:
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        if int(ID[0]) <= 447:
            dst_path = train_all_save_path + '/' + ID[0]  # Train_all
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
        else:
            dst_path = test_save_path + '/' + ID[0]  # Test
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


# train_val i.e. split train_all into train and val
for root, dirs, files in os.walk(train_all_save_path, topdown=True):
    for dir_name in dirs:
        for root_n, dirs_n, files_n in os.walk(train_all_save_path + '/' + dir_name):
            val_ind = np.random.randint(0, len(files_n))  # Take randomly
            val_file_name = files_n[val_ind]
            for name in files_n:
                if not name[-3:].lower() == ext:
                    continue
                ID = name.split('_')
                src_path = train_all_save_path + '/' + dir_name + '/' + name
                dst_path = train_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if name == val_file_name:
                    dst_path = val_save_path + '/' + ID[0]  # This image is used as val image
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)


# ------ Query and Gallery split from test data ------------------------------------------------------------------------
# A randomly chosen image is put in gallery folder (default) or the first image is put in gallery folder, and the rest
# are put in query folder. This helps to compare each query image to each gallery image and then evaluate using
# Cumulative Matching Characteristics (CMC-1 or rank-1 accuracy) i.e. top-1 accuracy (verification accuracy). Here you
# can generate query_i and gallery_i where i = 0, ..., 9 for 10 Monte Carlo runs for proper evaluations.

# Set N = 10 for Monte Carlo runs
N = 10
is_random_gallery = True
for i in range(N):
    query_i = 'query' + str(i)
    gallery_i = 'gallery' + str(i)
    query_save_path = save_path + '/' + query_i  # query0, query1, etc
    gallery_save_path = save_path + '/' + gallery_i  # gallery0, gallery1, etc
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(test_save_path, topdown=True):
        for dir_name in dirs:
            for root_n, dirs_n, files_n in os.walk(test_save_path + '/' + dir_name):
                val_ind = np.random.randint(0, len(files_n))
                val_file_name = files_n[val_ind]
                for name in files_n:
                    if not name[-3:].lower() == ext:
                        continue
                    ID = name.split('_')
                    src_path = test_save_path + '/' + dir_name + '/' + name
                    dst_path = query_save_path + '/' + ID[0]
                    if is_random_gallery:
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                        if name == val_file_name:
                            dst_path = gallery_save_path + '/' + ID[0]  # The gallery image is chosen randomly
                            if not os.path.isdir(dst_path):
                                os.mkdir(dst_path)
                    else:
                        if not os.path.isdir(dst_path):
                            os.mkdir(dst_path)
                            dst_path = gallery_save_path + '/' + ID[0]  # The first image is used as gallery image
                            os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

    # Copy the additional 211 (actually it is 213) identities into each gallery
    for root, dirs, files in os.walk(add2gallery_path, topdown=True):
        for name in files:
            if not name[-3:].lower() == ext:
                continue
            ID = name.split('_')
            ID_new = str(int(ID[0]) + 1000)  # Add new label to newly added subjects
            src_path = add2gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + ID_new  # Test
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
