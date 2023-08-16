"""
-*- coding: utf-8 -*-
@File  : weak_util.py
@author: Yaobaochen
@Time  : 2022/10/30 下午10:20
"""

import numpy as np
from math import ceil
import torch
from torch.nn import CrossEntropyLoss, Softmax, PairwiseDistance
import torch.nn.functional as F

CE = CrossEntropyLoss()
SM = Softmax(dim=1)
pdist = PairwiseDistance(p=2)

threshold = 0.75


def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def compute_feature_consistency_loss(logits, feat, logits_t, feat_t, target, cfg):
    loss = torch.mean(pdist(feat, feat_t))

    return loss


def compute_contrastive_loss(logits, feat, logits_t, feat_t, logits_other, target, cfg):
    temp = 10
    eps = 1e-10

    prob_t, label_t = SM(logits_t).max(dim=1)

    prob, label = SM(logits).max(dim=1)

    prob_other, label_other = SM(logits_other).max(dim=1)

    idx = (label_t == label_other) & (label_t == label) & (prob_t > threshold)

    idx_pos = torch.ones((feat.shape[0])).cuda() == 0
    for i in range(cfg.num_classes):
        idx_pos_class = idx & (label == i)
        if idx_pos_class.sum() < 1000:
            idx_pos = idx_pos | idx_pos_class
        else:
            prob_max, idx_max = prob_t.sort(descending=True)
            idx_pos[idx_max[idx_pos_class][0:1000]] = True

    stu = feat[idx_pos]
    pos = feat_t[idx_pos]
    pos_label = label[idx_pos]

    idx_choice = np.random.choice(range(feat_t.shape[0]), size=3000)
    cur_neg = feat_t[idx_choice]
    cur_label = label_t[idx_choice]

    pos_pesu = pos_label.unsqueeze(1).repeat(1, cur_neg.shape[0])
    neg_pesu = cur_label.unsqueeze(1).repeat(1, stu.shape[0])
    neg_mask = torch.ones((stu.shape[0], cur_neg.shape[0]), dtype=torch.float32).cuda()
    neg_mask *= (pos_pesu != neg_pesu.T)

    neg = (stu @ cur_neg.T) / temp
    down = (torch.exp(neg) * neg_mask).sum(-1)

    up = (stu * pos).sum(-1, keepdim=True) / temp
    up = torch.exp(up).squeeze(-1)

    loss = torch.mean(-torch.log(torch.clip(up / torch.clip(up + down, eps), eps)))

    if torch.isnan(loss):
        loss = 0

    return loss * 0.2


def update_prototype(prototype, logits_ta, feat_ta, logits_tb, feat_tb, cfg):

    prob_ta, label_ta = SM(logits_ta).max(dim=1)

    prob_tb, label_tb = SM(logits_tb).max(dim=1)

    idx = (label_ta == label_tb) & (prob_ta > threshold) & (prob_tb > threshold)

    for i in range(cfg.num_classes):
        idx_cls = (label_ta == i) & idx
        if idx_cls.sum() > 0:
            prob_ta_cls = prob_ta[idx_cls]
            prob_tb_cls = prob_tb[idx_cls]
            feat_ta_cls = feat_ta[idx_cls]
            feat_tb_cls = feat_tb[idx_cls]

            max_sort_ta = prob_ta_cls.sort(descending=True)[1]
            max_sort_tb = prob_tb_cls.sort(descending=True)[1]

            num = 500

            if max_sort_ta.shape[0] > num:
                feat_cls = feat_ta_cls[max_sort_ta[:num]]
            else:
                feat_cls = feat_ta_cls

            if max_sort_tb.shape[0] > num:
                feat_cls = torch.cat((feat_cls, feat_tb_cls[max_sort_tb[:num]]), 0)
            else:
                feat_cls = torch.cat((feat_cls, feat_tb_cls), 0)

            if cfg.is_prototype[i] == 0:
                prototype[i] = torch.mean(feat_cls, dim=0).unsqueeze(0)
                cfg.is_prototype[i] = 1
            else:
                prototype[i] = 0.75 * prototype[i] + 0.25 * torch.mean(feat_cls, dim=0).unsqueeze(0)

    # prototype = torch.where(torch.isnan(prototype), torch.full_like(prototype, 0), prototype)

    return prototype


def compute_prototype_loss(prototype, logits, feats, logits_ta, logits_tb, target, mask, cfg):
    temp = 0.5
    eps = 1e-10

    prob, label = SM(logits).max(dim=1)

    prob_ta, label_ta = SM(logits_ta).max(dim=1)

    prob_tb, label_tb = SM(logits_tb).max(dim=1)

    idx = (label_ta == label_tb) & (label == label_ta)
    feats = feats[idx]
    prob = prob[idx]
    label = label[idx]

    num = ceil(feats.shape[0] * 0.3)
    max_prob = prob.sort(descending=True)[1][0:num]
    feats = feats[max_prob]
    label = label[max_prob]

    # compute cosine similarity
    x = feats.unsqueeze(1)
    y = prototype.unsqueeze(0)
    output = F.cosine_similarity(x, y, dim=-1)
    prob_max, label_max = output.max(dim=1)

    with torch.no_grad():
        pos_mask = torch.zeros((output.shape[0] * output.shape[1]), dtype=torch.float32).cuda()
        row = torch.Tensor(range(output.shape[0])).int().cuda() * cfg.num_classes + label_max
        pos_mask[row] = 1
        pos_mask = pos_mask.reshape(output.shape[0], output.shape[1])
        neg_mask = (pos_mask == 0)

    pos = torch.sum(output * pos_mask, dim=1)
    up = torch.exp(pos / temp)

    down = (torch.exp(output / temp) * neg_mask).sum(-1)

    loss = torch.mean(-torch.log(torch.clip(up / torch.clip(up + down, eps), eps)))

    return loss
