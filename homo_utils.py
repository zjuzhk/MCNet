import random
import numpy as np
import cv2
import torch

def regenerate_homo_cv2(img1, large_img2, org_pts, dst_pts, homo_parameter):
    img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    large_img2 = torch.tensor(large_img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    differ = dst_pts - org_pts
    org_pts = torch.tensor([[0, 0], [127, 0], [0, 127], [127, 127]], dtype=torch.float32).unsqueeze(0)
    dst_pts = org_pts + differ
    H = cv2.getPerspectiveTransform(org_pts.squeeze().numpy(), dst_pts.squeeze().numpy())
    warp_img1 = np.transpose(cv2.warpPerspective(img1.squeeze().permute(1, 2, 0).numpy(), H, (128, 128)), (2, 0, 1))

    _, _, height, width = large_img2.shape
    marginal, perturb, patch_size = homo_parameter["marginal"], homo_parameter["perturb"], homo_parameter["patch_size"]
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    four_pts_perturb = []
    for i in range(4):
        t1 = random.randint(-perturb, perturb)
        t2 = random.randint(-perturb, perturb)
        four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])

    org_pts = np.array(four_pts, dtype=np.float32)
    dst_pts = np.array(four_pts_perturb, dtype=np.float32)
    H = torch.tensor(cv2.getPerspectiveTransform(org_pts, dst_pts), dtype=torch.float32).unsqueeze(0)
    warp_img2 = np.transpose(cv2.warpPerspective(large_img2.squeeze().permute(1, 2, 0).numpy(), H.squeeze().numpy(), (height, width)), (2, 0, 1))
    patch_img2 = torch.from_numpy(warp_img2[:, 32:160, 32:160])
    warp_img1 = torch.from_numpy(warp_img1)
    large_img2 = large_img2.squeeze()

    org_pts = torch.from_numpy(org_pts)
    dst_pts = torch.from_numpy(dst_pts)
    four_gt = dst_pts - org_pts

    return org_pts, dst_pts, four_gt, warp_img1, patch_img2, large_img2

def generate_homo(img1, img2, homo_parameter, transform=None):
    if transform is not None:
        img1, img2 = transform(image=img1)['image'], transform(image=img2)['image']
    img1, img2 = img1 / 255, img2 / 255 # normalize
    # define corners of image patch
    marginal, perturb, patch_size = homo_parameter["marginal"], homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    img1 = img1[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    img2 = img2[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    four_pts = four_pts - four_pts[np.newaxis, 0] + marginal # 将top_left设置为(marginal, marginal)
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb, perturb)
            t2 = random.randint(-perturb, perturb)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb // (i + 1)
            t2 = - perturb // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    patch_img1 = warped_img1[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    patch_img2 = img2[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    patch_img1 = torch.from_numpy(patch_img1).float().permute(2, 0, 1)
    patch_img2 = torch.from_numpy(patch_img2).float().permute(2, 0, 1)
    large_img1 = torch.from_numpy(warped_img1).float().permute(2, 0, 1)
    large_img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
    
    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, large_img1, large_img2 


def sequence_loss(four_preds, four_gt, args):
    """ Loss function defined over sequence of flow predictions """
    loss = 0
    loss = loss_function(four_preds, four_gt, args)
    mace = ((four_preds[-1] - four_gt)**2).sum(dim=-1).sqrt().mean(dim=-1).detach().cpu().numpy().mean()
    return loss, mace


def loss_function(four_points_pred, four_points_gt, args):
    loss = 0
    sp_flag = 0
    for i in range(len(four_points_pred)):
        x = torch.abs(four_points_pred[i] - four_points_gt).mean()
        if x < args.speed_threshold: sp_flag = 1
        if args.loss == 'l1':
            loss += x
        elif args.loss == 'l2':
            loss += x**2
        elif args.loss == 'speedupl1':
            loss += (x - sp_flag * (1/(x + args.epsilon)))

    return loss

def calculate_ace(four_preds, four_gt):
    ace = ((four_preds[-1] - four_gt)**2).sum(dim=-1).sqrt().mean(dim=-1).detach().cpu().numpy()
    return ace

