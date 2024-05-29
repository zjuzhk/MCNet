import numpy as np
import os, time, pprint
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import datasets
from network import *
from utils import *
from homo_utils import *
            
def test(args, glob_iter=None, homo_model=None):
    device = torch.device('cuda:'+ str(args.gpuid))
    test_loader = datasets.fetch_dataloader(args, split="test")
    if homo_model == None:
        homo_model = MCNet(args).to(device)
        if args.checkpoint is None:
            print("ERROR : no checkpoint")
            exit()
        state = torch.load(args.checkpoint, map_location='cpu')
        homo_model.load_state_dict(state['homo_model'])
        print("test with pretrained model")
    homo_model.eval()

    with torch.no_grad():
        mace_array = np.array([])
        for test_repeat in range(1): # repeat test multiple times to get stable result
            for i, data_batch in enumerate(test_loader):
                for key, value in data_batch.items(): 
                    if type(data_batch[key]) == torch.Tensor: data_batch[key] = data_batch[key].to(device)    
                pred_h4p_12 = homo_model(data_batch)
                # calculate metric
                mace = ((pred_h4p_12[-1] - data_batch["four_gt"])**2).sum(dim=-1).sqrt().mean(dim=-1).detach().cpu().numpy()
                mace_array = np.concatenate([mace_array, mace])
    
    print(f"mace:{round(mace_array.mean(), 3)}")
    if not args.nolog:
        visualize_predict_image(args, data_batch, pred_h4p_12, glob_iter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
    parser.add_argument('--gpuid', type=int, default=1)
    parser.add_argument('--note', type=str, default='', help='experiment notes')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='The log path')
    parser.add_argument('--nolog', action='store_true', default=False, help='save log file or not')
    parser.add_argument('--checkpoint', type=str, default="model.pth", help='Test model name')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--log_full_dir', type=str)
    parser.add_argument('--epsilon', type=float, default=0.1, help='loss parameter')
    parser.add_argument('--loss', type=str, default="speedup", help="speedup or l1 or l2 or speedupl1")
    parser.add_argument('--downsample', type=int, nargs='+', default=[4,2,1])
    parser.add_argument('--iter', type=int, nargs='+', default=[2,2,2])
    parser.add_argument('--speed_threshold', type=float, default=1, help='use speed-up when L1 < x')
    args = parser.parse_args()
    
    if not args.nolog:
        args.log_full_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + "_" + args.dataset + "_" + args.note)
        if not os.path.exists(args.log_full_dir): os.makedirs(args.log_full_dir)
        sys.stdout = Logger_(os.path.join(args.log_full_dir, f'record.log'), sys.stdout)
    pprint.pprint(vars(args))
    
    seed_everything(args.seed)
   
    test(args)

if __name__ == "__main__":
    main()
