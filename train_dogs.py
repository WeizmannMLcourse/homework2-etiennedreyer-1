import numpy as np
import torch
import torch.nn.functional as func
from tqdm.notebook import tqdm


def train_valid_loop(net, train_dl, valid_dl, Nepochs, learning_rate=0.001):

    train_loss = []
    valid_loss = []

    ### Optimizer
    optimizer = # use Adam

    ### Check for GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('Found GPU!')
        device = torch.device("cuda:0")

    net.to( # put it on the device

    for # loop over Nepochs

        ### Training
        net.train()

        train_loss_epoch = []
        for xb,yb in #which dataloader?
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.z#...please make sure the gradients are zeroed-out each time!
            pred = #pass the input through the net to get the prediction
            loss = func.#use the MSE loss between the prediction and the target
            # --> <-- ADD LINE TO DO BACKPROPAGATION HERE
            train_loss_epoch.append(loss.item())
            optimizer.s#...please make the optimizer step in the direction of negative gradient

        ### take the average of the loss over each batch and append it to the list
        train_loss.append(np.mean(train_loss_epoch))

        ### Validation
        net.eval()

        valid_loss_epoch = []
        for xb,yb in #which dataloader?
            xb = # move the validation input to the device
            yb = # move the validation target to the device
            pred = # same as in training loop
            loss = # same as in training loop
            valid_loss_epoch.append(loss.item())

        valid_loss.append(np.mean(valid_loss_epoch))

        ### Model checkpointing
        if epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), 'saved_model.pt')

        print('Epoch: ',epoch,' Train loss: ',train_loss[-1],' Valid loss: ',valid_loss[-1])

    #Bring net back to CPU
    net.cpu()

    return train_loss, valid_loss