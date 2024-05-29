import os, random, sys
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import math
import torch.nn as nn
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from kornia.geometry import get_perspective_transform, warp_perspective
import torchvision.transforms as transforms  

class Logger_(object):
    def __init__(self, filename=None, stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

    def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
            pass

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_input_image(args, data_batch):
    visualize_path = os.path.join(args.log_full_dir, "visualize", "input")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    
    torchvision.utils.save_image(data_batch["patch_img1_warp"], os.path.join(visualize_path, 'patch_img1_warp.jpg'))
    torchvision.utils.save_image(data_batch["patch_img2"], os.path.join(visualize_path, 'patch_img2.jpg'))
    torchvision.utils.save_image(data_batch["large_img1_warp"], os.path.join(visualize_path, 'large_img1_warp.jpg'))
    torchvision.utils.save_image(data_batch["large_img2"], os.path.join(visualize_path, 'large_img2.jpg'))
    
    org_pts = data_batch["org_pts"]
    org_pts = org_pts - org_pts[:, [0]]
    dst_pts = org_pts + data_batch["four_gt"]
    H_gt = get_perspective_transform(org_pts, dst_pts)
    patch_size = data_batch["patch_img1_warp"].shape[2]
    patch_img1_warp_check = warp_perspective(data_batch["patch_img1_warp"], H_gt, (patch_size, patch_size))
    torchvision.utils.save_image(patch_img1_warp_check, os.path.join(visualize_path, 'patch_img1_warp_check.jpg'))
    
def visualize_predict_image(args, data_batch, pred_h4p_12, iters):
    visualize_path = os.path.join(args.log_full_dir, "visualize", "output", "test")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    
    pred_h4p_12 = pred_h4p_12[-1]
    org_pts = data_batch["org_pts"]
    dst_pts = data_batch["dst_pts"]
    dst_pts_pred = data_batch["org_pts"] + pred_h4p_12
    H_pred = get_perspective_transform(org_pts, dst_pts_pred)
    large_size, patch_size = data_batch["large_img1_warp"].shape[2], data_batch["patch_img1_warp"].shape[2]
    large_img1_warp_predict = warp_perspective(data_batch["large_img1_warp"], H_pred, dsize=(large_size, large_size))
    torchvision.utils.save_image(large_img1_warp_predict, os.path.join(visualize_path, f'{iters}_large_img1_warp_predict.jpg'))
    
    patch_img1_warp_predict = torch.zeros((pred_h4p_12.shape[0], 3, patch_size, patch_size)).to(pred_h4p_12.device)
    top_left, bottom_right = [], []
    for idx in range(pred_h4p_12.shape[0]):
        top_left = org_pts[idx][0].cpu().numpy().astype(np.int32)
        bottom_right = org_pts[idx][3].cpu().numpy().astype(np.int32)
        patch_img1_warp_predict[idx] = large_img1_warp_predict[idx, :, top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
    torchvision.utils.save_image(patch_img1_warp_predict, os.path.join(visualize_path, f'{iters}_patch_img1_warp_predict.jpg'))
    torchvision.utils.save_image(data_batch["patch_img1_warp"], os.path.join(visualize_path, f'{iters}_patch_img1_warp.jpg'))
    torchvision.utils.save_image(data_batch["patch_img2"], os.path.join(visualize_path, f'{iters}_patch_img2.jpg'))
    
    org_pts = org_pts.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts = dst_pts.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts_pred = org_pts + pred_h4p_12.reshape(-1, 4, 1, 2).detach().cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    
    large_img1_warp = data_batch["large_img1_warp"].permute(0,2,3,1).cpu().numpy().copy()
    large_img2 = data_batch["large_img2"].permute(0,2,3,1).cpu().numpy().copy()
    
    for idx in range(pred_h4p_12.shape[0]):
        large_img1_warp[idx] = cv2.polylines(large_img1_warp[idx], np.int32([org_pts[idx]])     , True, (0,1,0), 3, cv2.LINE_AA)
        large_img2[idx]      = cv2.polylines(large_img2[idx]     , np.int32([dst_pts[idx]])     , True, (0,1,0), 3, cv2.LINE_AA)
        large_img2[idx]      = cv2.polylines(large_img2[idx]     , np.int32([dst_pts_pred[idx]]), True, (1,0,0), 2, cv2.LINE_AA)

    torchvision.utils.save_image(torch.from_numpy(large_img1_warp).permute(0,3,1,2), 
                                 os.path.join(visualize_path, f'{iters}_large_img1_warp_draw.jpg'))
    torchvision.utils.save_image(torch.from_numpy(large_img2).permute(0,3,1,2), 
                                 os.path.join(visualize_path, f'{iters}_large_img2_draw.jpg'))

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def correlation(fmap1, fmap2):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht*wd)
    fmap2 = fmap2.view(batch, dim, ht*wd) 

    corr = torch.matmul(fmap1.transpose(1,2), fmap2)
    corr = corr.view(batch, ht*wd, ht, wd)
    return corr  / torch.sqrt(torch.tensor(dim).float())

def exp_loss(pred, gt):
    return torch.mean(torch.exp(torch.abs(pred - gt) / 5))

def identityLoss(H, H_inv):
    batch_size = H.size()[0]
    Identity = torch.eye(3)
    if torch.cuda.is_available():
        Identity = Identity.to(H.device)
    Identity = Identity.unsqueeze(0).expand(batch_size,3,3)
    return F.l1_loss(H.bmm(H_inv), Identity)