from ast import iter_child_nodes
import torch
import torch.nn as nn
from update import *
from extractor import *
from corr import *
from utils import *
from flow_utils import *
from homo_utils import *

class MCNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fnet = BasicEncoder(output_dim=96, norm_fn='instance')
        self.update_blocks = nn.ModuleList([CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=4),
                                    CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=5),
                                    CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=6)])
                
        self.downsample = self.args.downsample
        self.iter = self.args.iter
        self.memory = {"deltaD":[], "scale":[], "delta_ace":[], "iteration":[]}
        self.instance_experts = []

    def forward(self, data_batch):
        image1, image2 = data_batch["patch_img1_warp"], data_batch["patch_img2"]
        
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        batch_size = image1.shape[0]
        four_point_disp = torch.zeros((batch_size, 2, 2, 2)).to(image1.device)
        four_point_predictions = []

        for downsample in self.downsample:
            idx = self.downsample.index(downsample)
            corr_fn = LocalCorr(fmap1[idx], fmap2[idx])
            coords0, _ = initialize_flow(image1, downsample=downsample)

            for _ in range(self.iter[idx]):
                coords1 = disp_to_coords(four_point_disp, coords0, downsample=downsample)              
                corr = corr_fn(coords1)   
                
                four_point_delta = self.update_blocks[idx](corr)
                four_point_disp =  four_point_disp + four_point_delta
                four_point_reshape = four_point_disp.permute(0,2,3,1).reshape(-1,4,2) # [top_left, top_right, bottom_left, bottom_right], [-1, 4, 2]
                
                four_point_predictions.append(four_point_reshape)
        
        return four_point_predictions

