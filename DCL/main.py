"""
-*- coding: utf-8 -*-
@File  : main.py
@author: Yaobaochen
@Time  : 2022/10/27 下午9:09
"""

import os
import sys
import glob
import time
import wandb
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter

from openpoints.utils import EasyConfig
from openpoints.dataset.data_util import voxelize
from openpoints.dataset import get_features_by_keys
from openpoints.loss import build_criterion_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious

from model.base_seg import BaseSeg
from utils.random_seed import seed_everything
from dataset.build_dataloader import build_dataloader
from utils.ckpt_util import save_checkpoint, load_checkpoint
from utils.weak_util import update_ema_variables, compute_feature_consistency_loss, compute_contrastive_loss, \
    update_prototype, compute_prototype_loss
from utils.data_perturbation import data_perturbation


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
           label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)

    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]  # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)  # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


@torch.no_grad()
def test(model, data_list, cfg):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    seed_everything(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.log_path + '/result'
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    for cloud_idx, data_path in enumerate(data_list):

        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        pbar = tqdm(range(len(idx_points)))

        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx + 1}-th cloud [{idx_subcloud}]/[{len_part}]]")

            idx_part = idx_points[idx_subcloud]
            coord_part = coord[idx_part]
            coord_part -= coord_part.min(0)

            feat_part = feat[idx_part] if feat is not None else None
            data = {'pos': coord_part}
            if feat_part is not None:
                data['x'] = feat_part
            if pipe_transform is not None:
                data = pipe_transform(data)
            if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                data['heights'] = torch.from_numpy(
                    coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
            if not cfg.dataset.common.get('variable', False):
                if 'x' in data.keys():
                    data['x'] = data['x'].unsqueeze(0)
                data['pos'] = data['pos'].unsqueeze(0)
            else:
                data['o'] = torch.IntTensor([len(coord)])
                data['batch'] = torch.LongTensor([0] * len(coord))

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            data['x'] = get_features_by_keys(data, cfg.feature_keys)

            logits, __ = model(data)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        # average merge overlapped multi voxels logits to original point set
        idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
        all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')

        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)

        if 'scannet' in cfg.dataset.common.NAME.lower():
            pred = pred.cpu().numpy().squeeze()
            label_int_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14,
                                 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
            pred = np.vectorize(label_int_mapping.get)(pred)
            save_file_name = data_path.split('/')[-1].split('_')
            save_file_name = save_file_name[0] + '_' + save_file_name[1] + '.txt'
            save_file_name = os.path.join(cfg.log_path+'/result/'+save_file_name)
            np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            logging.info(
                f'[{cloud_idx + 1}/{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}'
            )
            all_cm.value += cm.value

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, cm
    else:
        return None, None, None, None, None, None


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits, __ = model(data)

        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, accs = get_mious(tp, union, count)

    return miou, macc, oa, ious


def train_one_epoch(prototype, model_a, model_b, model_ta, model_tb, train_loader, criterion, optimizer_a, optimizer_b, scheduler_a, scheduler_b, scaler, epoch, cfg):

    loss_meter_a = AverageMeter()
    cm_a = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    loss_meter_b = AverageMeter()
    cm_b = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    model_a.train()  # set model to training mode
    model_b.train()
    model_ta.train()
    model_tb.train()

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)

        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        mask = torch.squeeze(data['mask'])

        data_perturb = data_perturbation(data, cfg)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits_a, feats_a = model_a(data_perturb)
            logits_b, feats_b = model_b(data)

            logits_a = logits_a.transpose(1, 2).reshape(-1, logits_a.shape[1])
            logits_b = logits_b.transpose(1, 2).reshape(-1, logits_b.shape[1])

            target = target.flatten()
            mask = mask.flatten()
            idx = mask == 1

            loss_sup_a = criterion(logits_a[idx], target[idx])
            loss_sup_b = criterion(logits_b[idx], target[idx])

        with torch.no_grad():
            logits_ta, feats_ta = model_ta(data)
            logits_tb, feats_tb = model_tb(data_perturb)
            logits_ta = logits_ta.transpose(1, 2).reshape(-1, logits_ta.shape[1])
            logits_tb = logits_tb.transpose(1, 2).reshape(-1, logits_tb.shape[1])

        loss_fc_a = compute_feature_consistency_loss(logits_a, feats_a, logits_ta, feats_ta, target, cfg)
        loss_fc_b = compute_feature_consistency_loss(logits_b, feats_b, logits_tb, feats_tb, target, cfg)

        loss_contra_a = compute_contrastive_loss(logits_a, feats_a, logits_tb, feats_tb, logits_ta, target, cfg)
        loss_contra_b = compute_contrastive_loss(logits_b, feats_b, logits_ta, feats_ta, logits_tb, target, cfg)

        with torch.no_grad():
            prototype = update_prototype(prototype, logits_ta, feats_ta, logits_tb, feats_tb, cfg)
        loss_pt_a = compute_prototype_loss(prototype, logits_a, feats_a, logits_ta, logits_tb, target, mask, cfg)
        loss_pt_b = compute_prototype_loss(prototype, logits_b, feats_b, logits_ta, logits_tb, target, mask, cfg)

        loss_a = loss_sup_a + loss_contra_a + loss_fc_a + loss_pt_a
        loss_b = loss_sup_b + loss_contra_b + loss_fc_b + loss_pt_b

        with open(config.log_path + '/loss.txt', 'a') as file:   # 打开文件
            file.write(f'loss_sup_a: {loss_sup_a:.3f},  loss_contra_a: {loss_contra_a:.3f},  loss_fc_a: {loss_fc_a:.3f},  loss_pt_a: {loss_pt_a:.3f},  '
                       f'loss_sup_b: {loss_sup_b:.3f},  loss_contra_b: {loss_contra_b:.3f},  loss_fc_b: {loss_fc_b:.3f},  loss_pt_b: {loss_pt_b:.3f}\n')

        if cfg.use_amp:
            scaler.scale(loss_a).backward()
            scaler.scale(loss_b).backward()
        else:
            loss_a.backward()
            loss_b.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), cfg.grad_norm_clip, norm_type=2)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), cfg.grad_norm_clip, norm_type=2)

            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer_a)
                scaler.step(optimizer_b)
                scaler.update()
            else:
                optimizer_a.step()
                optimizer_b.step()

            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler_a.step(epoch)
                scheduler_b.step(epoch)

        model_ta = update_ema_variables(model_ta, model_a, 0.99, epoch)
        model_tb = update_ema_variables(model_tb, model_b, 0.99, epoch)

        # update confusion matrix
        cm_a.update(logits_a.argmax(dim=1), target)
        loss_meter_a.update(loss_sup_a.item())

        cm_b.update(logits_b.argmax(dim=1), target)
        loss_meter_b.update(loss_sup_b.item())

        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}]")

    miou1, macc1, oa1, ious1, accs1 = cm_a.all_metrics()
    miou2, macc2, oa2, ious2, accs2 = cm_b.all_metrics()

    return (loss_meter_a.avg+loss_meter_b.avg)/2.0, (miou1+miou2)/2.0, (macc1+macc2)/2.0, (oa1+oa2)/2.0


def main(cfg):
    model_a = BaseSeg(cfg.model).cuda()
    model_b = BaseSeg(cfg.model).cuda()

    model_ta = BaseSeg(cfg.model).cuda()
    model_tb = BaseSeg(cfg.model).cuda()

    optimizer_a = build_optimizer_from_cfg(model_a, lr=cfg.lr, **cfg.optimizer)
    optimizer_b = build_optimizer_from_cfg(model_b, lr=cfg.lr, **cfg.optimizer)

    scheduler_a = build_scheduler_from_cfg(cfg, optimizer_a)
    scheduler_b = build_scheduler_from_cfg(cfg, optimizer_b)

    load_checkpoint(model_a, pretrained_path=cfg.mode_a_path)
    load_checkpoint(model_b, pretrained_path=cfg.mode_b_path)
    load_checkpoint(model_ta, pretrained_path=cfg.mode_a_path)
    load_checkpoint(model_tb, pretrained_path=cfg.mode_b_path)

    # build dataset
    val_loader = build_dataloader(cfg, 'val')
    train_loader = build_dataloader(cfg, 'train')

    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None

    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    prototype = torch.zeros((cfg.num_classes, 32)).cuda()
    cfg.is_prototype = torch.zeros((cfg.num_classes, 1), dtype=torch.int8)
    # ---------------------------------------------------------------------------------------------------------------- #
    #                                               Start train and val
    # ---------------------------------------------------------------------------------------------------------------- #
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start train and val >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    best_val, macc_when_best, oa_when_best, best_epoch = 0., 0., 0., 0

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        train_loss, train_miou, train_macc, train_oa = \
            train_one_epoch(prototype, model_a, model_b, model_ta, model_tb, train_loader, criterion, optimizer_a,
                            optimizer_b, scheduler_a, scheduler_b, scaler, epoch, cfg)

        is_best = False
        if cfg.min_val < epoch and epoch % cfg.val_freq == 0:
            val_miou_a, val_macc_a, val_oa_a, val_ious_a = validate(model_a, val_loader, cfg)
            val_miou_b, val_macc_b, val_oa_b, val_ious_b = validate(model_b, val_loader, cfg)
            if val_miou_a > best_val:
                is_best = True
                best_val = val_miou_a
                macc_when_best = val_macc_a
                oa_when_best = val_oa_a
                best_epoch = epoch
                logging.info(f'iou per: {val_ious_a}')

            save_checkpoint(cfg, model_a, epoch, optimizer_a, scheduler_a, additioanl_dict={'best_val': best_val}, is_best=is_best)
            if val_miou_b > best_val:
                is_best = True
                best_val = val_miou_b
                macc_when_best = val_macc_b
                oa_when_best = val_oa_b
                best_epoch = epoch
                logging.info(f'iou per: {val_ious_b}')
            save_checkpoint(cfg, model_b, epoch, optimizer_b, scheduler_b, additioanl_dict={'best_val': best_val}, is_best=is_best)

        lr = optimizer_a.param_groups[0]['lr']+optimizer_b.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} 'f'train_loss {train_loss:.2f}, train_miou {train_miou:.2f}'
                     f', best val miou {best_val:.2f}')

        if is_best:
            logging.info(
                f'Epoch {epoch} Find best ckpt, val_miou {best_val:.2f} '
                f'val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
            )

        if cfg.sched_on_epoch:
            scheduler_a.step(epoch)
            scheduler_b.step(epoch)

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Val End! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info(
        f'Best val @epoch{best_epoch} , val_miou {best_val:.2f}, val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}')
    if cfg.wandb:
        wandb.log({'val_miou': best_val, 'val_macc': round(macc_when_best, 2), 'val_oa': round(oa_when_best, 2)})

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                 Start testing
    # ---------------------------------------------------------------------------------------------------------------- #

    if cfg.test_mode == 1:
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start testing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Test model @epoch{best_epoch}, loading the ckpt......')

        model = BaseSeg(cfg.model)
        model.cuda()

        load_checkpoint(model, pretrained_path=os.path.join(cfg.log_path, 'best.pth'))

        test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, cfg.test_data_list, cfg)

        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Result on area 5 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'Test result @epoch{best_epoch}: test_oa {test_oa:.2f}, '
                         f'test_macc {test_macc:.2f}, test_miou {test_miou:.2f}')
            logging.info(f'iou per: {test_ious}')

        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished all !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if cfg.wandb:
            wandb.log({'test_miou': test_miou, 'test_macc': test_macc, 'test_oa': test_oa})
    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":

    # load config
    config = EasyConfig()

    config.load("cfg_s3dis.yaml", recursive=True)

    config.seed = np.random.randint(1, 10000)
    seed_everything(config.seed)

    # create log dir
    config.log_path = './log/' + config.dataset.common.NAME + '-seed_' + str(config.seed) + '-' + time.strftime(
        '%Y.%m.%d-%H:%M:%S')
    os.makedirs(config.log_path)
    if 's3dis' in config.dataset.common.NAME.lower():
        os.system('cp %s %s' % ("cfg_s3dis.yaml", config.log_path))

    # create logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        handlers=[
            logging.FileHandler('%s/%s.log' % (
                config.log_path,
                config.dataset.common.NAME + '-seed' + str(config.seed) + '-' + time.strftime('%Y%m%d-%H%M'))),
            logging.StreamHandler(sys.stdout)
        ]
    )

    f = open(config.log_path + '/loss.txt', 'w')

    if config.wandb:
        wandb.init(project="ssl_seg", name='train')

    if 's3dis' in config.dataset.common.NAME.lower():
        raw_root = os.path.join(config.dataset.common.data_root, config.dataset.common.NAME, 'raw')
        data_list = sorted(os.listdir(raw_root))
        config.test_data_list = [os.path.join(raw_root, item) for item in data_list if
                                 'Area_{}'.format(config.dataset.common.test_area) in item]

    main(config)
