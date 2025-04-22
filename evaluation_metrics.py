import numpy as np
import torch


# Evaluate
def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)

    # Compute Cosine similarity score of a query feature with each of gallery image features.
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # Predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # Good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index  # In our case!

    CMC_tmp = compute_mAP(index, good_index)

    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # Find good_index index
    n_good = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(n_good):
        d_recall = 1.0 / n_good
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def compute_CMC_mAP(result):

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = np.asarray(result['query_label'])
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = np.asarray(result['gallery_label'])

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label)  # Average CMC
    mAP = ap/len(query_label)  # Mean Average Precision

    return CMC, mAP
