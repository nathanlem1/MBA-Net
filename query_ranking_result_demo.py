import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import scipy.io
import torch
from torchvision import datasets
matplotlib.use('agg')


# Evaluate
parser = argparse.ArgumentParser(description='Demo of retrieving ranked results given an index of a query image.')
parser.add_argument('--query_index', default=115, type=int, help='test_image_index, eg. 0 - 971 for dorsal right, '
                                                                 '0 - 988 for dorsal left, 0 - 917 for palmar right, '
                                                                 '0 - 948 for palmar left, or 0 - 1992 for HD')
parser.add_argument('--test_dir', default='./11k/train_val_test_split_dorsal_r', type=str,
                    help='Test dir path: '
                         './11k/train_val_test_split_dorsal_r' './11k/train_val_test_split_dorsal_l'
                         './11k/train_val_test_split_palmar_r'  './11k/train_val_test_split_palmar_l'  # For 11k
                         './HD/Original Images/train_val_test_split')  # For HD
parser.add_argument('--top_k', default=5, type=int, help='Top k images are retrieved and shown i.e. top k similar'
                                                         'images to the query image e.g. 5, 10, etc.')
opts = parser.parse_args()

data_dir = opts.test_dir
gallery = 'gallery0_all'  # 'gallery0_all' for 11k data set.
# gallery = 'gallery0'  # 'gallery0' for HD data set.
query = 'query0'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [gallery, query]}


# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


# Load saved files - choose this correctly!
result = scipy.io.loadmat('result_mba_dr.mat')   # For 11k dorsal right
# result = scipy.io.loadmat('result_mba_dl.mat')  # For 11k dorsal left
# result = scipy.io.loadmat('result_mba_pr.mat')  # For 11k palmar right
# result = scipy.io.loadmat('result_mba_pl.mat')  # For 11k palmar left
# result = scipy.io.loadmat('result_mba_hd.mat')  # For HD


query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


# Sort the images
def sort_img(qf, gf):
    query = qf.view(-1, 1)

    # Compute Cosine similarity score of a query feature with each of gallery image features.
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # Predict index
    index = np.argsort(score)  # From small to large
    index = index[::-1]   # From large to small

    return index


i = opts.query_index
top_k = opts.top_k
index = sort_img(query_feature[i], gallery_feature)

# Visualize the rank result
query_path, _ = image_datasets[query].imgs[i]
query_label = query_label[i]
print(query_path)
print('Top %s images are as follow:' % (top_k))
try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, top_k + 1, 1)
    ax.axis('off')
    imshow(query_path, 'query')
    for i in range(top_k):
        ax = plt.subplot(1, top_k + 1, i + 2)
        ax.axis('off')
        img_path, _ = image_datasets[gallery].imgs[index[i]]
        label = gallery_label[index[i]]
        img = plt.imread(img_path)
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d' % (i+1), color='green')
            rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=2, edgecolor='green', fill=False)
        else:
            ax.set_title('%d' % (i+1), color='red')
            rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=2, edgecolor='red', fill=False)
        ax.add_patch(rect)
        print(img_path)
except RuntimeError:
    for i in range(top_k):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
