import torch
import torchgeometry as tgm
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def initialize_flow(img, downsample=4):
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//downsample, W//downsample).to(img.device)
    coords1 = coords_grid(N, H//downsample, W//downsample).to(img.device)

    return coords0, coords1


def disp_to_coords(four_point, coords, downsample=4):
    four_point = four_point / downsample
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([coords.shape[3]-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, coords.shape[2]-1])
    four_point_org[:, 1, 1] = torch.Tensor([coords.shape[3]-1, coords.shape[2]-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(coords.shape[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    H = tgm.get_perspective_transform(four_point_org, four_point_new)
    gridy, gridx = torch.meshgrid(torch.linspace(0, coords.shape[3]-1, steps=coords.shape[3]), torch.linspace(0, coords.shape[2]-1, steps=coords.shape[2]))
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, coords.shape[3] * coords.shape[2]))),
                       dim=0).unsqueeze(0).repeat(coords.shape[0], 1, 1).to(four_point.device)
    points_new = H.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    coords = torch.cat((points_new[:, 0, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1),
                      points_new[:, 1, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1)), dim=1)
    return coords
