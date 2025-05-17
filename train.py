import os, time, pprint
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import optim
import datasets
from network import *
from utils import *
from homo_utils import *


def train(args):
    device = torch.device('cuda:'+ str(args.gpuid))
    train_loader = datasets.fetch_dataloader(args, split="train")
    homo_model = MCNet(args).to(device)

    if args.checkpoint is not None:
        print("Load pretrained checkpoint: ", *args.checkpoint)
        state = torch.load(args.checkpoint, map_location='cpu')
        homo_model.load_state_dict(state['homo_model'])

    homo_model.train()
    print(f"{round(count_parameters(homo_model)/1000000, 2)}M parameters")
    optimizer = optim.AdamW(homo_model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    train_mace, glob_iter = 0, 0
    start_time = time.time()
    
    while glob_iter <= args.num_steps:
        for i, data_batch in enumerate(train_loader):
            end_time = time.time() # calculate time remaining
            time_remain = (end_time - start_time) * (args.num_steps - glob_iter)
            start_time = time.time()
            
            if glob_iter == 0 and not args.nolog: visualize_input_image(args, data_batch)
            for key, value in data_batch.items(): data_batch[key] = data_batch[key].to(device)
            
            optimizer.zero_grad()
            pred_h4p_12 = homo_model(data_batch)

            loss, mace = sequence_loss(pred_h4p_12, data_batch["four_gt"], args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(homo_model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            
            # calculate metric
            train_mace += mace
            if glob_iter % args.print_freq == 0 and glob_iter != 0:
                print("Training: Iter[{:0>3}]/[{:0>3}] mace: {:.3f} lr={:.8f} time: {:.2f}h".format(glob_iter, args.num_steps, train_mace / args.print_freq, scheduler.get_lr()[0], time_remain/3600))
                train_mace = 0

            # save model
            if glob_iter % args.val_freq == 0 and glob_iter != 0:
                filename = 'model' + '_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(args.log_full_dir, filename)
                print("save model to: ", model_save_path)
                checkpoint = {"homo_model": homo_model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict()}
                torch.save(checkpoint, model_save_path)
                args.checkpoint = model_save_path
            
            if glob_iter % args.val_freq == 0 and glob_iter != 0:
                homo_model.eval()
                test(args, glob_iter, homo_model)
                homo_model.train()

            glob_iter += 1
            if glob_iter > args.num_steps: break

            
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
        
    
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--note', type=str, default='', help='experiment notes')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='The log path')
    parser.add_argument('--nolog', action='store_true', default=False, help='save log file or not')
    parser.add_argument('--checkpoint', type=str, default=None, help='Test model name')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--log_full_dir', type=str)
    parser.add_argument('--epsilon', type=float, default=0.1, help='loss parameter')
    parser.add_argument('--downsample', type=int, nargs='+', default=[4,2,1])
    parser.add_argument('--iter', type=int, nargs='+', default=[2,2,2])
    parser.add_argument('--exp', type=str, default='', help='experiment description')
    parser.add_argument('--loss', type=str, default="speedupl1", help="l1 or l2 or speedupl1")
    parser.add_argument('--speed_threshold', type=float, default=1, help='use speed-up when L1 < x')
    args = parser.parse_args()
    
    return args
            
def main():
    args = parse()
    
    if not args.nolog:
        args.log_full_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + "_" + args.dataset + "_" + args.note)
        if not os.path.exists(args.log_full_dir): os.makedirs(args.log_full_dir)
        sys.stdout = Logger_(os.path.join(args.log_full_dir, f'record.log'), sys.stdout)
    pprint.pprint(vars(args))
    
    seed_everything(args.seed)
   
    train(args)

if __name__ == "__main__":
    main()
