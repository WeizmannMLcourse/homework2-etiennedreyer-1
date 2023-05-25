import torch
import numpy as np
from custom_dataset import CustomDataset
from model_denoise import DenoiseNet
from torch.utils.data import Dataset, DataLoader
import sys
import glob
import torch.nn as nn

def evaluate_on_dataset(files=None):

    if files==None:
        files = glob.glob('dog-breeds/rottweiler/*')

    test_ds = CustomDataset(files)
    dataloader = DataLoader(test_ds,batch_size=16)

    net = DenoiseNet()

    net.load_state_dict(torch.load('saved_model.pt',map_location=torch.device('cpu')))

    net.eval()

    loss_func = nn.MSELoss()

    loss = 0
    n_batches = 0
    with torch.no_grad():
        for x,y in dataloader:
			
            n_batches+=1
			
            pred = net(x)
			
            loss+= loss_func(pred,y).item()

    return loss/n_batches


if __name__ == "__main__":

    files = sys.argv[1] if len(sys.argv) > 1 else None
    loss = evaluate_on_dataset(files)

    print(loss)

