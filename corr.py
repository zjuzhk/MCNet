import torch.nn as nn
import torch.nn.functional as F
from flow_utils import *
import corr_implement

try:
    import alt_cuda_corr
except:
    pass # alt_cuda_corr is not compiled
    
class LocalCorr:
    def __init__(self, fmap1, fmap2):
        self.map1 = fmap1
        self.map2 = fmap2
        self.N, self.C, self.H, self.W = fmap1.shape
        self.coords = coords_grid(self.N, self.H, self.W).to(fmap1.device)

    def warp(self, coords, image, h, w):
        coords[: ,0 ,: ,:] = 2.0 *coords[: ,0 ,: ,:].clone() / max(self.W -1 ,1 ) -1.0
        coords[: ,1 ,: ,:] = 2.0 *coords[: ,1 ,: ,:].clone() / max(self.H -1 ,1 ) -1.0

        coords = coords.permute(0 ,2 ,3 ,1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
        return output

    def __call__(self, coords):
        map2_warp = self.warp(coords, self.map2, self.H, self.W)
        corr = corr_implement.FunctionCorrelation(self.map1, map2_warp)
        return corr
    



