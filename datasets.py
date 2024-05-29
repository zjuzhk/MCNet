import torch
from torch.utils.data import Dataset, DataLoader
import os, json, cv2
from glob import glob
import matplotlib.pyplot as plt
from homo_utils import generate_homo


class MSCOCO(Dataset):
    def __init__(self, split):
        self.homo_parameter = {"marginal":32, "perturb":32, "patch_size":128}
        
        if split == 'train':
            root_img2 = '/home/csy/datasets/csy/mscoco/train2017'
            root_img1 = '/home/csy/datasets/csy/mscoco/train2017'        
        else:
            root_img2 = '/home/csy/datasets/mscoco/test2017'
            root_img1 = '/home/csy/datasets/mscoco/test2017'

        self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))    

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = cv2.resize(img1, (320, 240))
        img2 = cv2.resize(img2, (320, 240))
            
        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape
        
        patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2 = generate_homo(img1, img2, homo_parameter=self.homo_parameter, transform=None)

        return {"patch_img1_warp":patch_img1_warp, "patch_img2":patch_img2, "four_gt":four_gt,
                "org_pts":org_pts, "dst_pts":dst_pts,
                "large_img1_warp":large_img1_warp, "large_img2":large_img2}

class GoogleEarth(Dataset):
    def __init__(self, split='train'):
        
        self.split = split
        self.homo_parameter = {"marginal": 32, "perturb": 32, "patch_size": 128}
        if split == 'train':
            self.img1_path = '/datasets/GoogleEarth/train2014_template/'
            self.img2_path = '/datasets/GoogleEarth/train2014_input/'
            self.label_path = '/datasets/GoogleEarth/train2014_label/'
        else:
            self.img1_path = '/home/csy/datasets/csy/GoogleEarth/val2014_template/'
            self.img2_path = '/home/csy/datasets/csy/GoogleEarth/val2014_input/'
            self.label_path = '/home/csy/datasets/csy/GoogleEarth/val2014_label/'

        self.img_name = os.listdir(self.img1_path)

    def __len__(self):  
        return len(self.img_name)

    def __getitem__(self, index):  

        patch_img1_warp = plt.imread(self.img1_path + self.img_name[index])
        large_img2 = plt.imread(self.img2_path + self.img_name[index])
        patch_img1_warp, large_img2 = patch_img1_warp / 255, large_img2 / 255

        with open(self.label_path + self.img_name[index].split('.')[0] + '_label.txt', 'r') as outfile:
            data = json.load(outfile)

        top_left = [data['location'][0]['top_left_u'], data['location'][0]['top_left_v']]
        top_right = [data['location'][1]['top_right_u'], data['location'][1]['top_right_v']]
        bottom_left = [data['location'][2]['bottom_left_u'], data['location'][2]['bottom_left_v']]
        bottom_right = [data['location'][3]['bottom_right_u'], data['location'][3]['bottom_right_v']]
        
        org_pts = torch.tensor([[32,32], [159,32], [32,159], [159,159]], dtype=torch.float32)
        dst_pts = torch.tensor([top_left, top_right, bottom_left, bottom_right], dtype=torch.float32)
        four_gt = dst_pts - org_pts

        patch_img1_warp = torch.tensor(patch_img1_warp, dtype=torch.float32).permute(2, 0, 1)
        large_img2 = torch.tensor(large_img2, dtype=torch.float32).permute(2, 0, 1)
        patch_img2 = large_img2[:, 32:160, 32:160]

        return {"patch_img1_warp":patch_img1_warp, "patch_img2":patch_img2, "four_gt":four_gt,
                "org_pts":org_pts, "dst_pts":dst_pts,
                "large_img1_warp":torch.ones_like(large_img2, dtype=torch.float32), "large_img2":large_img2}

class homo_dataset(Dataset):
    def __init__(self, split, dataset, args):
        self.dataset = dataset
        self.args = args
        self.homo_parameter = {"marginal":32, "perturb":32, "patch_size":128}
        
        if split == 'train':
            if dataset == 'ggmap':
                root_img1 = '/datasets/GoogleMap/train2014_input'
                root_img2 = '/datasets/GoogleMap/train2014_template_original'    
            if dataset == 'spid':
                root_img1 = '/datasets/moving_object/img_pair_train_new/img1'
                root_img2 = '/datasets/moving_object/img_pair_train_new/img2'
                
        else:
            if dataset == 'ggmap':
                root_img1 = '/home/csy/datasets/csy/GoogleMap/val2014_input'
                root_img2 = '/home/csy/datasets/csy/GoogleMap/val2014_template_original'
            if dataset == 'spid':
                root_img1 = '/home/csy/datasets/csy//moving_object/img_pair_test_new/img1'
                root_img2 = '/home/csy/datasets/csy//moving_object/img_pair_test_new/img2'
                
        self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))                

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img_size = self.homo_parameter["patch_size"] + 2 * self.homo_parameter["marginal"]
        img1 = cv2.resize(img1, (img_size, img_size))
        img2 = cv2.resize(img2, (img_size, img_size))
            
        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape
        
        patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2 = generate_homo(img1, img2, homo_parameter=self.homo_parameter, transform=None)

        return {"patch_img1_warp":patch_img1_warp, "patch_img2":patch_img2, "four_gt":four_gt,
                "org_pts":org_pts, "dst_pts":dst_pts,
                "large_img1_warp":large_img1_warp, "large_img2":large_img2}


def fetch_dataloader(args, split='test'):

    if args.dataset == "googleearth": dataset = GoogleEarth(split=split)
    elif args.dataset == "mscoco": dataset = MSCOCO(split=split)
    else: dataset = homo_dataset(split=split, dataset=args.dataset, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=16, drop_last=False)
    print('Test with %d image pairs' % len(dataset))

    return dataloader

