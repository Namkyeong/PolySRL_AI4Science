# from data import Custom_Data
import argument
import random
import os

import numpy as np
import torch

torch.set_num_threads(2)
os.environ['OMP_NUM_THREADS'] = "2"

def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args, unknown = argument.parse_args()
    seed_everything(0)
    
    dataset = torch.load("./dataset/processed/pretrain_data_multi.pt")
    
    if args.embedder == 'PolySRL':
        from models import PolySRL_Trainer
        embedder = PolySRL_Trainer(args, dataset)
    
    if args.embedder == 'PolySRL_det':
        # This is for ablation studies, 
        # i.e., use mean vector for soft contrastive learning without probabilistic embeddings.
        from models import PolySRL_det_Trainer
        embedder = PolySRL_det_Trainer(args, dataset)
    
    if args.embedder == '3DInfomax':
        from models import Infomax_Trainer
        embedder = Infomax_Trainer(args, dataset)

    if args.embedder == 'MPBandG':
        from models import MPBandG_Trainer
        embedder = MPBandG_Trainer(args, dataset)
    
    if args.embedder == 'MPFormE':
        from models import MPFormE_Trainer
        embedder = MPFormE_Trainer(args, dataset)
    
    if args.embedder == 'GraphCL':
        from models import GraphCL_Trainer
        embedder = GraphCL_Trainer(args, dataset)

    embedder.train()



if __name__ == "__main__":
    main()