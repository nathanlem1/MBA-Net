import argparse
import copy
import os
import time
from shutil import copyfile

import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import yaml

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss
from lr_scheduler import LRScheduler
from model.MBA import ResNet50_MBA
from random_erasing import RandomErasing

version = torch.__version__


def set_seed(seed):
    """ For reproducibility """

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
          'support (https://github.com/NVIDIA/apex)')


# Train model
def train_model(model, criterion, lr_scheduler, optimizer, args, data_loaders, num_epochs=70):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # ---------------------------------
        lr = lr_scheduler.update(epoch)  # Warmup strategy is included.
        optimizer.param_groups[0]['lr'] = 0.1 * lr  # For pretrained layers
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
                if now_batch_size < args.batch_size:  # skip the last batch
                    continue

                if torch.cuda.is_available():
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
                    loss += criterion(outputs['x'][i + 1], labels)
                    score += sm(outputs['x'][i + 1])

                _, preds = torch.max(score.data, 1)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    if args.fp16:  # We use optimizer to backward loss
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

            epoch_loss = running_loss / len(data_loaders[phase].dataset.samples)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset.samples)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            # Save intermediate models, and training curve
            if phase == 'val':
                if epoch+1 % 10 == 0:
                    save_network(model, epoch+1, args)
                draw_curve(epoch, args)

            # Deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:  # > is changed to >=
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best', args)
    return model


# Draw training curves
def draw_curve(current_epoch, args):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(args.f_name, args.m_name, 'training_curve.jpg'))


# Save model
def save_network(network, epoch_label, args):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(args.f_name, args.m_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(args.gpu_ids[0])


# Options
def main():
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
                             './model_11k_d_r'  './model_11k_d_l'  './model_11k_p_r'  './model_11k_p_l')  # For 11k
    parser.add_argument('--data_type', default='11k', type=str, help='Data type: 11k or HD')
    parser.add_argument('--m_name', default='ResNet50_MBA', type=str,
                        help='Output model name - ResNet50_MBA for ResNet50 with MBA model.')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')  # 10, 20, 32, etc
    parser.add_argument('--color_jitter', action='store_true', default=True, help='use color jitter in training')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--ls', action='store_true', default=True, help='Use label smoothing with cross entropy.')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='learning rate for new parameters, try 0.015, 0.02, 0.05, For pretrained parameters, it '
                             'is 10 times smaller than this')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use: 0, 8, etc. Setting to 8 workers may run faster.')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use: sgd or adam')
    parser.add_argument('--part_h', default=1, type=int,
                        help='Number of horizontal partitions e.g. 1,2, .... Default is 1')
    parser.add_argument('--part_v', default=1, type=int,
                        help='Number of vertical partitions e.g. 1,2, ... Default is 1')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention (both spatial and channel).')
    parser.add_argument('--relative_pos', action='store_true', default=True,
                        help='Use relative positional encodings.')
    parser.add_argument('--is_repr', action='store_true', default=True,
                        help='For reproducibility during experimentation, for instance, for hyper-parameters tuning.')
    parser.add_argument('--fp16', action='store_true',
                        help='Use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--gpu_ids', default='0', type=str, help='Which GPUs to use - gpu_ids: e.g. 0  0,1,2  0,2')
    args = parser.parse_args()

    # For reproducibility
    if args.is_repr:
        seed = 3  # You can set the seed to any fixed value.
        set_seed(seed)
    else:
        cudnn.benchmark = True  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the
        # fastest. For PyTorch reproducibility, you need to set to False at cost of slightly lower run-time performance
        # but easy for experimentation.

    # Create folder
    if not os.path.isdir(args.f_name):
        os.mkdir(args.f_name)

    # For GPUs
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    args.gpu_ids = gpu_ids  # Update GPU IDs

    # Set gpu ids. You can check how many GPUs are there using torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(gpu_ids) > 0 and device == 'cuda':
        torch.cuda.set_device(gpu_ids[0])
        # cudnn.benchmark = True

    # Load Data
    transform_train_list = [
        transforms.Resize((356, 356), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((324, 324)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize((324, 324), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    if args.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=args.erasing_p, mean=[0.0, 0.0, 0.0])]

    if args.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + \
                               transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    train_all = ''
    if args.train_all:
        train_all = '_all'

    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'train' + train_all),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val'])

    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=False)
                    for x in ['train', 'val']}

    class_names = image_datasets['train'].classes
    args.num_classes = len(class_names)

    # Instantiate the model
    model = ResNet50_MBA(args.num_classes, relative_pos=args.relative_pos, part_h=args.part_h, part_v=args.part_v,
                         use_attention=args.use_attention)

    print(model)

    # Optimizer - giving greater lr to newly added layers gives slightly better result.
    if hasattr(model, 'backbone'):
        base_param_ids = set(map(id, model.backbone.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr_mult': 0.1*args.lr},
            {'params': filter(lambda p: p.requires_grad, new_params), 'lr_mult': 1.0}]
        # param_groups = [
        #     {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': 0.1*args.lr},
        #     {'params': filter(lambda p: p.requires_grad, new_params), 'lr': args.lr}]
    else:
        param_groups = model.parameters()

    if args.optimizer == 'sgd':
        optimizer_ft = torch.optim.SGD(param_groups, lr=args.lr,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
    elif args.optimizer == 'adam':
        optimizer_ft = torch.optim.Adam(
            param_groups, lr=args.lr,
            weight_decay=5e-4,
        )
    else:
        raise ValueError('Set the optimizer to either sgd or adam')

    # Learning rate scheduler
    lr_scheduler = LRScheduler(base_lr=0.0008, step=[40, 60],
                               factor=0.5, warmup_epoch=10,
                               warmup_begin_lr=0.000008)

    # Create a folder to save the trained model in
    dir_name = os.path.join(args.f_name, args.m_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Record every run
    copyfile('./train.py', dir_name+'/train.py')

    # Save args
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # Send model to GPU; it is recommended to use DistributedDataParallel, instead of DataParallel, to do multi-GPU
    # training, even if there is only a single node.
    # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model = model.cuda()

    # Use fp16
    if args.fp16:
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

    # Define loss function
    if args.ls:
        criterion = LabelSmoothingCrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    #  For drawing curves
    global y_loss, y_err, x_epoch, fig, ax0, ax1
    y_loss = dict()  # Loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = dict()
    y_err['train'] = []
    y_err['val'] = []
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    # Train and save the best model
    model = train_model(model, criterion, lr_scheduler, optimizer_ft, args, data_loaders, num_epochs=70)  # 60, 70, 200


# Execute from the interpreter
if __name__ == "__main__":
    main()
