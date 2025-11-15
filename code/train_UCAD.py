import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (SlicGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, val_2d

from config import parse_args, patients_to_slices


cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
# /home/chengboding/data/ACDC  /home/chengboding/data/synapse 
parser.add_argument('--root_path', type=str, default='/home/chengboding/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Synapse/UCAD', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--test_interval', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

# slic add
parser.add_argument('--num_labels', type=int,  default=20, help='num_labels')
parser.add_argument('--skip_pretrain', type=int,  default=0, help='skip pretraining or not')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')

parser.add_argument('--train_list', type=str, default='train_BCP.list', help='train_list') 
parser.add_argument('--test_list', type=str, default='test_BCP.list', help='test_list')
parser.add_argument('--unc_weight', type=float, default=0.2, help='weight of loss unc')

args = parser.parse_args()
dice_loss = None
CustomDataset = None


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_2DLargestCC(segmentation, class_num, skip_classes=[]):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, class_num + 1):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()

            if c in skip_classes:
                class_list.append(temp_prob * c)
                continue

            labels = label(temp_prob)          
            
            if np.max(labels) != 0:
                counts = np.bincount(labels.flat)[1:]
                largestCC = labels == (np.argmax(counts) + 1)
                class_list.append(largestCC * c) 
            else:
                class_list.append(temp_prob) 
        
        n_batch = np.sum(class_list, axis=0)
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_masks(output, class_num, nms=0, skip_classes=[]):
    probs = F.softmax(output, dim=1)
    _, pred_mask = torch.max(probs, dim=1)
    
    if nms == 1:
        pred_mask = get_2DLargestCC(pred_mask, class_num, skip_classes)
    
    return pred_mask

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


_centroid_cache = {}
def get_cutmix_mask_from_centroids_torch(segments: torch.Tensor, cases: list, num_labels: int):
    device = segments.device
    B, H, W = segments.shape
    mask_batch = torch.ones((B, H, W), dtype=torch.uint8, device=device) 

    segments_np = segments.cpu().numpy()

    for b in range(B):
        case_id = cases[b]
        seg = segments_np[b]

        if case_id not in _centroid_cache:
            labels = np.unique(seg)
            centroids = {}
            for lbl in labels:
                coords = np.argwhere(seg == lbl)
                if coords.size > 0:
                    cy, cx = coords.mean(axis=0)  # (y, x)
                    centroids[lbl] = (cy, cx)
            _centroid_cache[case_id] = centroids

        centroids = _centroid_cache[case_id]
        labels = list(centroids.keys())

        if num_labels == -1:
            n_select = np.random.randint(1, len(labels))
        else:
            n_select = min(num_labels, len(labels))

        start_label = np.random.choice(labels)
        start_centroid = np.array(centroids[start_label])

        distances = {
            lbl: np.linalg.norm(np.array(centroids[lbl]) - start_centroid)
            for lbl in labels
        }

        selected_labels = sorted(distances, key=distances.get)[:n_select]

        mask_np = ~np.isin(seg, selected_labels)  # bool: True=1, False=0
        mask_batch[b] = torch.from_numpy(mask_np.astype(np.uint8)).to(device)

    return mask_batch

def get_cutmix_mask_uncertainty(
    segments: torch.Tensor,           
    cases: list,                 
    prob_maps: torch.Tensor,        
    num_labels: int,           
    temperature: float = 0.5   
):
    device = segments.device
    B, H, W = segments.shape
    mask_batch = torch.ones((B, H, W), dtype=torch.uint8, device=device)

    entropy_map = -torch.sum(prob_maps * torch.log(prob_maps + 1e-8), dim=1)  # [B,H,W]

    segments_np = segments.detach().cpu().numpy()
    entropy_np = entropy_map.detach().cpu().numpy()

    for b in range(B):
        case_id = cases[b]
        seg = segments_np[b]
        ent = entropy_np[b]

        if case_id not in _centroid_cache:
            labels = np.unique(seg)
            centroids = {}
            for lbl in labels:
                coords = np.argwhere(seg == lbl)
                if coords.size > 0:
                    cy, cx = coords.mean(axis=0)
                    centroids[lbl] = (cy, cx)
            _centroid_cache[case_id] = centroids
        centroids = _centroid_cache[case_id]
        labels = np.array(list(centroids.keys()))

        sp_unc = np.zeros(len(labels), dtype=np.float32)
        for i, lbl in enumerate(labels):
            sp_unc[i] = ent[seg == lbl].mean() if (seg == lbl).any() else 0.0

        def _norm(x):
            xmin, xmax = float(x.min()), float(x.max())
            if xmax - xmin < 1e-12:
                return np.zeros_like(x, dtype=np.float32)
            return (x - xmin) / (xmax - xmin)

        unc_n = _norm(sp_unc)
        T = max(temperature, 1e-6)
        logits = unc_n / T
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        if num_labels == -1:
            n_select = int(np.random.randint(1, len(labels)))
        else:
            n_select = int(min(num_labels, len(labels)))

        selected_labels = np.random.choice(labels, size=n_select, replace=False, p=probs)

        mask_np = ~np.isin(seg, selected_labels)
        mask_batch[b] = torch.from_numpy(mask_np.astype(np.uint8)).to(device)

    return mask_batch




def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def mix_loss_entropy(output, mixed_label, mask, teacher_logits, l_weight=1.0, u_weight=0.5, unlab=False, beta=0.5):
    CE = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.DiceLoss(n_classes=args.num_classes)
    mixed_label = mixed_label.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, mixed_label.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, mixed_label.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, mixed_label) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, mixed_label) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    
    EPS = 1e-6
    # student probs + entropy
    p_s = F.softmax(output, dim=1)
    H_s = -torch.sum(p_s * torch.log(p_s + EPS), dim=1, keepdim=True)
    # teacher probs + entropy
    p_t = F.softmax(teacher_logits, dim=1)
    H_t = -torch.sum(p_t * torch.log(p_t + EPS), dim=1, keepdim=True)

    exp_H_s = torch.exp(beta * H_s)
    exp_H_t = torch.exp(beta * H_t)

    if unlab:
        unlabel_mask = mask
    else:
        unlabel_mask = 1 - mask

    # mask: only compute on unlabeled region
    unlabel_mask = (1 - mask).unsqueeze(1)   # (B,1,H,W,[D])

    diff = (p_s - p_t) ** 2 / (exp_H_s + exp_H_t)
    diff = diff * unlabel_mask               # mask out labeled

    H_s_masked = H_s * unlabel_mask
    H_t_masked = H_t * unlabel_mask

    # final loss
    valid_voxels = unlabel_mask.sum() + 1e-16
    loss_unc = (diff.sum(dim=1).sum() + beta * (H_s_masked + H_t_masked).sum()) / valid_voxels
    
    return loss_dice, loss_ce, loss_unc


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
    

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CustomDataset(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([SlicGenerator(args.patch_size)]),
                            train_list=args.train_list,
                            test_list=args.test_list
                            )
    db_val = CustomDataset(base_dir=args.root_path, split="val",
                          train_list=args.train_list,
                          test_list=args.test_list
                            )
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.exp, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            segments_batch = sampled_batch['segments']
            segments_batch = segments_batch.cuda()
            cases_batch = sampled_batch['case']

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            seg_a, seg_b = segments_batch[:labeled_sub_bs], segments_batch[labeled_sub_bs:args.labeled_bs]
            case_a, case_b = cases_batch[:labeled_sub_bs], cases_batch[labeled_sub_bs:args.labeled_bs]
                
            img_mask = get_cutmix_mask_from_centroids_torch(seg_a, case_a, num_labels=args.num_labels)  

            loss_mask = img_mask.clone()
            img_mask = img_mask.unsqueeze(1)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num > 0 and iter_num % args.test_interval == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1), metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i+1), metric_list[class_i, 3], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_jc = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = CustomDataset(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([SlicGenerator(args.patch_size)]),
                            train_list=args.train_list,
                            test_list=args.test_list
                            )
    db_val = CustomDataset(base_dir=args.root_path, split="val",
                          train_list=args.train_list,
                          test_list=args.test_list
                            )

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.exp,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    iter_num = 0
    epoch_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            segments_batch = sampled_batch['segments']
            segments_batch = segments_batch.cuda()
            cases_batch = sampled_batch['case']

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            # ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            seg_a, seg_b = segments_batch[:labeled_sub_bs], segments_batch[labeled_sub_bs:args.labeled_bs]
            seg_ua, seg_ub = segments_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], segments_batch[args.labeled_bs + unlabeled_sub_bs:]
            case_a, case_b = cases_batch[:labeled_sub_bs], cases_batch[labeled_sub_bs:args.labeled_bs]
            case_ua, case_ub = cases_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], cases_batch[args.labeled_bs + unlabeled_sub_bs:]
            # print(seg_a.shape)
            with torch.no_grad():
                logits = model(img_a)
                prob_maps = torch.softmax(logits, dim=1)
                logits_ub = model(uimg_b)
                prob_maps_ub = torch.softmax(logits_ub, dim=1)
            img_mask_a = get_cutmix_mask_uncertainty(seg_a, case_a, prob_maps, num_labels=args.num_labels, temperature=args.temperature)
            img_mask_ub = get_cutmix_mask_uncertainty(seg_ub, case_ub, prob_maps_ub, num_labels=args.num_labels, temperature=args.temperature)

            loss_mask_a = img_mask_a.clone()
            img_mask_a = img_mask_a.unsqueeze(1)
            loss_mask_ub = img_mask_ub.clone()
            img_mask_ub = img_mask_ub.unsqueeze(1)
            
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                if 'Synapse' == args.exp.split("/")[0]:
                    plab_a = get_masks(pre_a, args.num_classes, nms=1)
                    plab_b = get_masks(pre_b, args.num_classes, nms=1)
                else:
                    plab_a = get_masks(pre_a, args.num_classes, nms=1)
                    plab_b = get_masks(pre_b, args.num_classes, nms=1)

            net_input_unl = uimg_a * img_mask_a + img_a * (1 - img_mask_a)
            net_input_l = img_b * img_mask_ub + uimg_b * (1 - img_mask_ub)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)

            mixed_unl = plab_a * loss_mask_a + lab_a * (1 - loss_mask_a) 
            mixed_l = lab_b * loss_mask_ub + plab_b * (1 - loss_mask_ub)

            beta = losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch, max_beta=2.0, min_beta=0.5)

            unl_dice, unl_ce, unc_1 = mix_loss_entropy(out_unl, mixed_unl, loss_mask_a, pre_a, u_weight=args.u_weight, unlab=True, beta=beta)
            l_dice, l_ce, unc_2 = mix_loss_entropy(out_l, mixed_l, loss_mask_ub, pre_b, u_weight=args.u_weight, beta=beta)

            loss_ce = unl_ce + l_ce 
            loss_dice = unl_dice + l_dice
            loss_unc = unc_1 + unc_2

            loss = (loss_dice + loss_ce) / 2  + loss_unc * args.unc_weight       

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num) 

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f, unc: %f'%(iter_num, loss, loss_dice, loss_ce, loss_unc))

            if iter_num > 0 and iter_num % args.test_interval == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                print(metric_list)
                logging.info(metric_list)
                for class_i in range(num_classes-1):
                    # dice, jc, hd95, asd
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1), metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i+1), metric_list[class_i, 3], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_jc = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    CustomDataset = parse_args(args)
    dice_loss = losses.DiceLoss(n_classes=args.num_classes)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)

    # -- path to save models
    pre_snapshot_path = "../model/{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "../model/{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(__file__, os.path.dirname(self_snapshot_path))

    if args.skip_pretrain:
        print("Skip pre-training, start self-training directly.")
    else:
        # Pre_train
        logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


