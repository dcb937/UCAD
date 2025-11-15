import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
from medpy import metric
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio
import glob
from scipy.ndimage import zoom

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

from dataloaders.dataset import (ACDCDataSets, SynapseDataSets, SlicGenerator, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

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
parser.add_argument('--exp', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--ckpt', type=str, default='', help='ckpt')
parser.add_argument('--model', type=str, default='unet', help='model_name')

args = parser.parse_args()

CustomDataset = None

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    # 检查预测和真实标签是否都包含正样本
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        jc = 0
        # asd = 0
        hd95 = 0
        return dice, jc, hd95, asd
    else: 
        return 0, 0, 0, 0
    
def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    print('xxx')
    for i in range(1, classes):
        print(f'classes: {i}')
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

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


class MCNetWrapper(nn.Module):
    """MCNet模型包装器，只返回第一个输出"""
    def __init__(self, mcnet_model):
        super(MCNetWrapper, self).__init__()
        self.mcnet_model = mcnet_model
    
    def forward(self, x):
        # MCNet返回两个输出，我们只使用第一个
        output1, output2 = self.mcnet_model(x)
        return output1

def evaluate(args, ckpt_path):
    num_classes = args.num_classes
    
    # 根据模型名称选择合适的模型初始化方式
    model_name = os.path.basename(ckpt_path).replace('.pth', '').lower()
    print(f"Model name: {model_name}")
    
    if 'mcnet' in model_name:
        # MCNet模型使用特殊的初始化方式
        base_model = net_factory(net_type='mcnet2d_v1', in_chns=1, class_num=num_classes)
        # 使用包装器只返回第一个输出
        model = MCNetWrapper(base_model)
    elif 'cml' in model_name:
        model = net_factory(net_type='unet_cml', in_chns=1, class_num=num_classes)
    elif 'urpc' in model_name:
        model = net_factory(net_type='unet_urpc', in_chns=1, class_num=num_classes)
    else:
        # 其他模型使用BCP_net
        model = BCP_net(in_chns=1, class_num=num_classes)
    
    saved_state_dict = torch.load(ckpt_path)

    # 3. 将加载的字典加载到你的模型中
    if 'mcnet' in model_name:
        # 对于MCNet，需要加载到包装器内的基础模型
        model.mcnet_model.load_state_dict(saved_state_dict)
    else:
        # 其他模型直接加载
        model.load_state_dict(saved_state_dict)
    db_val = CustomDataset(base_dir=args.root_path, split="val",
                          train_list=args.train_list,
                          test_list=args.test_list
                            )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    model.eval()
    metric_list = 0.0
    for _, sampled_batch in enumerate(valloader):
        print('aa')
        metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_val)
    print(metric_list)

    performance = np.mean(metric_list, axis=0)[0]
    mean_jc = np.mean(metric_list, axis=0)[1]
    mean_hd95 = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]

    model_name = os.path.basename(ckpt_path).replace('.pth', '')
    print('Model: {} | dice: {:.4f}, jc: {:.4f}, hd95: {:.4f}, asd: {:.4f}'.format(
        model_name, performance, mean_jc, mean_hd95, mean_asd))
    
    return {
        'model': model_name,
        'dice': performance,
        'jc': mean_jc,
        'hd95': mean_hd95,
        'asd': mean_asd
    }


def evaluate_all_models(args):
    """依次加载并验证ckpt目录下的所有模型"""
    ckpt_dir = '/data/chengboding/SUMix/code/ckpt_synapse_1'
    
    # 获取所有.pth文件
    model_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    
    if not model_files:
        print(f"在目录 {ckpt_dir} 中未找到.pth模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    print("-" * 60)
    
    # 存储所有结果
    all_results = []
    
    # 依次验证每个模型
    for i, model_file in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] 正在验证模型: {os.path.basename(model_file)}")
        print("-" * 40)
        
        try:
            result = evaluate(args, model_file)
            all_results.append(result)
        except Exception as e:
            print(f"验证模型 {os.path.basename(model_file)} 时出错: {str(e)}")
            continue
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("所有模型验证结果汇总:")
    print("=" * 60)
    print(f"{'模型名称':<15} {'Dice':<10} {'JC':<10} {'HD95':<10} {'ASD':<10}")
    print("-" * 60)
    
    for result in all_results:
        print(f"{result['model']:<15} {result['dice']:<10.4f} {result['jc']:<10.4f} "
              f"{result['hd95']:<10.4f} {result['asd']:<10.4f}")
    
    # 找出最佳模型
    if all_results:
        best_dice_model = max(all_results, key=lambda x: x['dice'])
        print("\n" + "=" * 60)
        print("最佳模型 (基于Dice分数):")
        print(f"模型: {best_dice_model['model']}")
        print(f"Dice: {best_dice_model['dice']:.4f}")
        print(f"JC: {best_dice_model['jc']:.4f}")
        print(f"HD95: {best_dice_model['hd95']:.4f}")
        print(f"ASD: {best_dice_model['asd']:.4f}")


if __name__ == "__main__":
    CustomDataset = parse_args(args)
    
    # 检查是否指定了单个模型还是批量验证
    if hasattr(args, 'ckpt') and args.ckpt:
        # 验证单个模型
        print(f"验证单个模型: {args.ckpt}")
        evaluate(args, args.ckpt)
    else:
        # 批量验证所有模型
        evaluate_all_models(args)

    


