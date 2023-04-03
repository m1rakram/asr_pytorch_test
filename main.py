import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
import pandas as pd
import torch_optimizer

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from my_datasets import COMMONVOICE

# project imports
from my_datasets import TrainDataset, TestDataset, mel_len, preprocess_data, transform_tr
from model import QuartzNet
from train_test import train


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == '__main__':
    BATCH_SIZE = 80
    NUM_EPOCHS = 5
    N_MELS     = 64

    set_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Loading data and loaders
    train_dataset = COMMONVOICE(transforms=transform_tr)
    test_dataset = COMMONVOICE(transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=preprocess_data, drop_last=True,
                          num_workers=0, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=preprocess_data, drop_last=True,
                          num_workers=0, pin_memory=True)

    ### wandb logins


    ### Creating melspecs on GPU
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,           ### 22050, 48000
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS                ### 64,    80
    ).to(device)
    # with augmentations
    
    
    melspec_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256,  n_mels=N_MELS),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35),
    ).to(device)

    ### Creating model from scratch
    model = QuartzNet(n_mels=64, num_classes=28)
    model.to(device)

    opt = torch_optimizer.NovoGrad(
                        model.parameters(),
                        lr=0.01,
                        betas=(0.8, 0.5),
                        weight_decay=0.001,
    )
    scheduler  = CosineAnnealingLR(opt, T_max=50, eta_min=0, last_epoch=-1)
    CTCLoss = nn.CTCLoss(blank=0).to(device)
    train(model, opt, train_loader, scheduler, CTCLoss, device,
                      n_epochs=NUM_EPOCHS, val_dl=val_loader,
                      melspec=melspec, melspec_transforms=melspec_transforms)

