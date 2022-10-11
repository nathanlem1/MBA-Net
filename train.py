import argparse
import copy
import os
import sys
import time
from shutil import copyfile

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import yaml

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

# Adding Folder MBA to the system path. Note that a module is just a Python program that ends with .py extension and a
# folder that contains a module becomes a package.
sys.path.insert(0, './MBA')
from MBA import MBA
# from MBA.MBA import MBA

from label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss
from lr_scheduler import LRScheduler
from random_erasing import RandomErasing

version = torch.__version__


# For reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)            # For Numpy reproducibility; this is only used in case any library depends on numpy!
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Use fp16 for faster training with low precision
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda '
          'support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

# Options
parser = argparse.ArgumentParser(description='Training MBA-Net with ResNet50 as a backbone network for hand-based '
                                             'person recognition (identification) as part of a H-Unique project.')
parser.add_argument('--data_dir',
                    default='./11k/train_val_test_split_dorsal_r',
                    type=str, help='Training dir path: ' 
                            './HD/Original Images/train_val_test_split'  # For HD
                            './11k/train_val_test_split_dorsal_r' './11k/train_val_test_split_dorsal_l'
                            './11k/train_val_test_split_palmar_r' './11k/train_val_test_split_palmar_l')  # For 11k
parser.add_argument('--f_name', default='./model_11k_d_r', type=str,
                    help='Output folder name - ./model_HD or '  # For HD dataset
                         './model_11k_d_r'  './model_11k_d_l'  './model_11k_p_r'  './model_11k_p_l')  # For 11k dataset
parser.add_argument('--data_type', default='11k', type=str, help='Data type: hd or 11k')
parser.add_argument('--m_name', default='ResNet50_MBA1', type=str,
                    help='Output model name - ResNet50_MBA for ResNet50 with MBA model.')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')  # 10, 20, 32, etc
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--ls', action='store_true', help='Use label smoothing with cross entropy.')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate for new parameters, try 0.015, 0.02, 0.05, '
                                                           'For pretrained parameters, it is 10 times smaller than this')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use: sgd or adam')
parser.add_argument('--use-biDir-relation', action='store_true', help='Use bi-directional relation in addition to the '
                                                                      'original features.')
parser.add_argument('--relative_pos', action='store_true', help='Use relative positional encodings.')
parser.add_argument('--attention_fn', type=str, default='abd_fn',
                    help='Attention type for MBA: abd_fn (default) or rga_fn.')
parser.add_argument('--part_h', default=1, type=int, help='Number of horizontal partitions e.g. 1,2, .... Default is 1')
parser.add_argument('--part_v', default=1, type=int, help='Number of vertical partitions e.g. 1,2, ... Default is 1')
parser.add_argument('--use_attention', action='store_true', help='Use attention (both spatial and channel).')
parser.add_argument('--is_repr', action='store_true',
                    help='For reproducibility during experimentation, for instance, for hyper-parameters tuning.')
parser.add_argument('--fp16', action='store_true',
                    help='Use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--gpu_ids', default='0', type=str, help='Which GPUs to use - gpu_ids: e.g. 0  0,1,2  0,2')
opt = parser.parse_args()

# We set these conditions for simplicity
opt.ls = True  # Set to True to use label smoothing with cross entropy.
opt.color_jitter = True  # Set to True to use color jitter.
opt.is_repr = True  # Set to True for reproducibility.
opt.use_biDir_relation = True  # Set to True for using bi-directional relation in addition to the feature itself.
opt.relative_pos = True  # Use relative positional encodings
opt.use_attention = True  # Use attention (both spatial and channel)

# For reproducibility
if opt.is_repr:
    seed = 3  # You can set the seed to any fixed value.
    set_seed(seed)
else:
    cudnn.benchmark = True  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    # For PyTorch reproducibility, you need to set to False at cost of slightly lower run-time performance but easy for
    # experimentation.

fp16 = opt.fp16
data_dir = opt.data_dir
m_name = opt.m_name
f_name = opt.f_name
if not os.path.isdir(f_name):
    os.mkdir(f_name)
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# Set gpu ids
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check how many GPUs are there using torch.cuda.device_count()
if len(gpu_ids) > 0 and device == 'cuda':
    torch.cuda.set_device(gpu_ids[0])
    # cudnn.benchmark = True

# Load Data

transform_train_list = [
    transforms.Resize((356, 356), interpolation=3),
    transforms.RandomCrop((324, 324)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    transforms.Resize(size=(324, 324), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + \
                           transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = dict()
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all), data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                               shuffle=True, num_workers=8, pin_memory=False)  # 8 workers work faster
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(data_loaders['train']))
print(time.time()-since)


# Training the model
# ---------------------------------------------------------------------------------------------------------------------
# Now, let's write a general function to train a model. Here, we will illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter 'scheduler' is an LR scheduler object from 'torch.optim.lr_scheduler'.
# ----------------------------------------------------------------------------------------------------------------------

y_loss = dict()  # Loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = dict()
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # ---------------------------------
        lr = lr_scheduler.update(epoch)  # Warmup strategy is included.
        optimizer.param_groups[0]['lr'] = 0.1*lr  # For pretrained layers
        optimizer.param_groups[1]['lr'] = lr  # For new layers
        # --------------------------------
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in data_loaders[phase]:
                # Get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batch_size:  # skip the last batch
                    continue

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)  # Softmax to convert the features to scaled positive values
                loss = 0
                score = 0
                for i in range(len(outputs['x'][0])):  # The 1st part (Local-Aware)
                    loss += criterion(outputs['x'][0][i], labels)
                    score += sm(outputs['x'][0][i])
                for i in range(len(outputs['x']) - 1):  # The 2nd part (Global + Attention)
                    loss += criterion(outputs['x'][i+1], labels)
                    score += sm(outputs['x'][i+1])

                _, preds = torch.max(score.data, 1)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    if fp16:  # We use optimizer to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                # Statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            # # ---------------------------
            # if phase == 'train':
            #     scheduler.step()
            # # --------------------------

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)

            # Save intermediate models, and training curve
            if phase == 'val':
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

            # Deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:   # > is changed to >=
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    return model
# ----------------------------------------------------------------------------------------------------------------------


# Draw Curve
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(f_name, m_name, 'training_curve.jpg'))


# Save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(f_name, m_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


# Fine-tuning the ConvNet
model = MBA(len(class_names), use_biDir_relation=opt.use_biDir_relation, attention_fn=opt.attention_fn,
            relative_pos=opt.relative_pos, part_h=opt.part_h, part_v=opt.part_v, use_attention=opt.use_attention)

opt.num_classes = len(class_names)

print(model)


# Optimizer
if hasattr(model, 'backbone'):    # Why using 'backbone' gives slightly better result for MBA?
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr_mult': 1.0},
        {'params': filter(lambda p: p.requires_grad, new_params), 'lr_mult': 1.0}]
    # param_groups = [
    #     {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': 0.1*opt.lr},
    #     {'params': filter(lambda p: p.requires_grad, new_params), 'lr': opt.lr}]
else:
    param_groups = model.parameters()


if opt.optimizer == 'sgd':
    optimizer_ft = torch.optim.SGD(param_groups, lr=opt.lr,   # lr is reset again to opt.lr equally
                                   momentum=0.9,
                                   weight_decay=5e-4,
                                   nesterov=True)
elif opt.optimizer == 'adam':
    optimizer_ft = torch.optim.Adam(
        param_groups, lr=opt.lr, # lr is reset again to opt.lr equally! Remove lr=opt.lr to use the previously set value
        weight_decay=5e-4,
    )
else:
    raise NameError


# Learning rate scheduler
lr_scheduler = LRScheduler(base_lr=0.0008, step=[40, 60],   # Standard (ours)
                           factor=0.5, warmup_epoch=10,
                           warmup_begin_lr=0.000008)


# Train and evaluate
# Create a folder to save the trained model in
dir_name = os.path.join(f_name, m_name)   # ./model_HD or ./model_knuckle
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# Record every run
copyfile('./train.py', dir_name+'/train.py')
# copyfile('./model.py', dir_name+'/model.py')

# Save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# Send model to GPU; it is recommended to use DistributedDataParallel, instead of DataParallel, to do multi-GPU
# training, even if there is only a single node.
# model = torch.nn.DataParallel(model, device_ids=[0]).cuda()  # Doesn't work during testing at the moment!
model = model.cuda()

# Use fp16
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

# # Decay LR by a factor of 0.1 every 30 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)  # None
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones = [20, 40], gamma=0.1) # With default lr = 0.0003
exp_lr_scheduler = None  # lr_scheduler is called in train_method function.

# Define loss function
if opt.ls:
    criterion = LabelSmoothingCrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()


# Train and save the best model
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=70)  # 60, 70, 600, 200


