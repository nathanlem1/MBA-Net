import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import scipy.io
import yaml

from evaluation_metrics import compute_CMC_mAP
from model.MBA import ResNet50_MBA

try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda '
          'support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


# Load model
def load_network(network, opt):
    save_path = os.path.join(opt.f_name, opt.m_name, 'net_%s.pth' % opt.which_epoch)  # Make sure which model to use!
    network.load_state_dict(torch.load(save_path))
    return network


# Flip image
def fliplr(img):
    """
    Flip horizontal
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# Extract feature from a trained model.# Extract feature from  a trained model.
def extract_feature(model, data_loaders, opt):
    features = torch.FloatTensor()
    count = 0
    for data in data_loaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)

        if opt.part_h * opt.part_v > 1 and opt.use_attention:  # All Parts + Global + Attention features are used.
            features_dim = 2048 * (3 + opt.part_h * opt.part_v)
        elif opt.part_h * opt.part_v > 1 and not opt.use_attention:  # Attention features are not used!
            features_dim = 2048 * (1 + opt.part_h * opt.part_v)
        elif opt.part_h * opt.part_v <= 1 and opt.use_attention:  # Part (local) features are not used!
            features_dim = 2048 * 3
        else:  # if opt.part_h * opt.part_v <= 1 and not opt.use_attention
            features_dim = 2048  # Use global features only

        ff = torch.FloatTensor(n, features_dim).zero_().cuda()  # For 2048-D
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = img.cuda()

            outputs = model(input_img)
            output_ffs = outputs['features']  # Gives better result on 324 x 324 on 11k-Dr. # Todo: ?
            if opt.part_h * opt.part_v > 1 and opt.use_attention:
                output_ffs0 = output_ffs[0].view(output_ffs[0].shape[0], -1)
                output_ffs = torch.cat((output_ffs0, output_ffs[1], output_ffs[2], output_ffs[3]),
                                       1)  # For using all
                # components
            elif opt.part_h * opt.part_v > 1 and not opt.use_attention:
                output_ffs0 = output_ffs[0].view(output_ffs[0].shape[0], -1)
                output_ffs = torch.cat((output_ffs0, output_ffs[1]), 1)  # Attention features are not used.
            elif opt.part_h * opt.part_v <= 1 and opt.use_attention:  # Part (local) features are not used!
                output_ffs = torch.cat((output_ffs[1], output_ffs[2], output_ffs[3]), 1)
            else:
                output_ffs = output_ffs[1]  # Use global features only

            ff += output_ffs

        # Normalize feature
        ff_norm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(ff_norm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


# Get ids
def get_id(img_path):
    labels = []
    for path, v in img_path:
        # filename = os.path.basename(path)
        # label = filename.split('_')[0]
        label = path.split('/')[-2]  # For 11k hand data set, on Linux
        # label = path.split('\\')[-2]  # For 11k hand data set, on Windows
        labels.append(int(label))
    return labels


def main():
    parser = argparse.ArgumentParser(description='Testing a trained ResNet50 with MBA model for hand identification as '
                                                 'part of a H-Unique project.')
    parser.add_argument('--test_dir',
                        default='./11k/train_val_test_split_dorsal_r',
                        type=str,
                        help=' Path to test_data: '
                             './11k/train_val_test_split_dorsal_r'  './11k/train_val_test_split_dorsal_l'
                             './11k/train_val_test_split_palmar_r'  './11k/train_val_test_split_palmar_l'  # For 11k
                             './HD/Original Images/train_val_test_split')  # For HD
    parser.add_argument('--f_name', default='./model_11k_d_r', type=str,
                        help='Output folder name - '
                             './model_11k_d_r  ./model_11k_d_l  ./model_11k_p_r  ./model_11k_p_l'  # For 11k
                             'or ./model_HD'   # For HD
                             'Note: Adjust the data-type in opts.yaml when evaluating cross-domain performance.')
    parser.add_argument('--m_name', default='ResNet50_MBA', type=str,
                        help='Saved model name - ResNet50_MBA for ResNet50 with MBA model.')
    parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or best')
    parser.add_argument('--batch_size', default=14, type=int, help='batch_size')  # 256, 40
    parser.add_argument('--fp16', action='store_true', help='use fp16.')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')

    # Args
    opt = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # Load the training config
    config_path = os.path.join(opt.f_name, opt.m_name, 'opts.yaml')

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    opt.fp16 = config['fp16']
    opt.data_type = config['data_type']
    opt.part_h = config['part_h']
    opt.part_v = config['part_v']
    opt.use_attention = config['use_attention']
    opt.relative_pos = config['relative_pos']

    if 'num_classes' in config:
        opt.num_classes = config['num_classes']  # The number of classes the model is trained on!
    else:
        opt.num_classes = 251  # 410

    str_ids = opt.gpu_ids.split(',')
    m_name = opt.m_name
    f_name = opt.f_name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # Set GPU ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # Load Data: We will use torchvision and torch.utils.data packages for loading the data, with appropriate transforms.
    data_transforms = transforms.Compose([
        transforms.Resize((324, 324), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = test_dir

    # Load Collected data Trained model
    print('-------Test has started ------------------')

    model_structure = ResNet50_MBA(opt.num_classes, relative_pos=opt.relative_pos, part_h=opt.part_h, part_v=opt.part_v,
                                   use_attention=opt.use_attention)

    model = load_network(model_structure, opt)

    # Send model to GPU; it is recommended to use DistributedDataParallel, instead of DataParallel, to do multi-GPU
    # training, even if there is only a single node.
    model = model.eval()
    if use_gpu:
        # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model = model.cuda()

    # For N = 10 Monte Carlo runs
    # You can change the data_type in opts.yaml if you want to perform cross-domain performance evaluation!
    if opt.data_type == '11k':
        galleries = ['gallery0_all', 'gallery1_all', 'gallery2_all', 'gallery3_all', 'gallery4_all', 'gallery5_all',
                     'gallery6_all', 'gallery7_all', 'gallery8_all', 'gallery9_all']  # For 11k
    elif opt.data_type == 'HD':
        galleries = ['gallery0', 'gallery1', 'gallery2', 'gallery3', 'gallery4', 'gallery5', 'gallery6', 'gallery7',
                     'gallery8', 'gallery9']   # For HD
    else:
        print('Please choose the correct data type!')
        exit()

    queries = ['query0', 'query1', 'query2', 'query3', 'query4', 'query5', 'query6', 'query7', 'query8', 'query9']

    CMC_total = 0
    mAP_total = 0
    for i in range(len(galleries)):
        g = galleries[i]
        q = queries[i]

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in [g, q]}
        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                                       shuffle=False, num_workers=8) for x in [g, q]}

        gallery_path = image_datasets[g].imgs
        query_path = image_datasets[q].imgs

        gallery_label = get_id(gallery_path)
        query_label = get_id(query_path)

        # Extract feature
        with torch.no_grad():
            gallery_feature = extract_feature(model, data_loaders[g], opt)
            query_feature = extract_feature(model, data_loaders[q], opt)

        # Save to Matlab for check
        result_fl = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
                     'query_f': query_feature.numpy(), 'query_label': query_label}
        scipy.io.savemat('result.mat', result_fl)

        print(m_name)
        result = '%s/%s/result.txt' % (f_name, m_name)
        # os.system('python3 compute_accuracy.py | tee -a %s' % result_file)
        # os.system('python3 compute_CMC_mAP.py | tee -a %s' % result)

        CMC_i, mAP_i = compute_CMC_mAP(result_fl)
        CMC_total += CMC_i
        mAP_total += mAP_i

        print('Result on %s and %s is:' % (g, q))
        print('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC_i[0], CMC_i[4], CMC_i[9], mAP_i))

    # Mean over N = 10 Monte Carlo runs
    CMC = CMC_total/len(galleries)
    mAP = mAP_total/len(galleries)

    print('\nMean result over N = %s Monte Carlo runs is:' % (len(galleries)))
    print('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

    res = open(result, 'w')
    res.write('Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

    # # Save for the later plot
    # res_cmc = '%s/%s/CMC_gpa.npy' % (f_name, m_name)  # CMC_gpa.npy, CMC_res50.npy, CMC_vgg.npy
    # np.save(res_cmc, CMC)

    print('-----Test is done!------------------')


# Execute from the interpreter
if __name__ == "__main__":
    main()
