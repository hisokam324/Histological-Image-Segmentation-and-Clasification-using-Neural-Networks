import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import json

# Definimos un Dataset de Pytorch
class SignalDataset(Dataset):
    def __init__(self, X, Y):
        if not torch.is_tensor(X):
            self.X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            self.Y = torch.Tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def select_device():
    # Eligiendo dispositivo (se puede cambiar en notebook settings)
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU disponible")
    else:
        device = torch.device("cpu")
        print("GPU no disponible, usando CPU")
    
    return device

def load_images(DATA_PATH = "data_y", getMask = True):

    with open('configuration.json') as file:
        configuration = json.load(file)

    IMG_HEIGHT = configuration["image"]["heigth"]
    IMG_WIDTH = configuration["image"]["width"]
    IMG_CHANNELS = configuration["image"]["channels"]

    IMAGE_PATH = DATA_PATH + "/image/"
    
    image_ids = sorted(os.listdir(IMAGE_PATH))
    
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    print("imagenes")
    for n in tqdm(range(len(image_ids)), total=len(image_ids)):
        img = imread(IMAGE_PATH + image_ids[n])
        X[n] = img
    
    X = X.transpose(0, 3, 1, 2)

    if getMask:
        print("mascaras")
        MASK_PATH = DATA_PATH + "/mask/"
        mask_ids = sorted(os.listdir(MASK_PATH))
        Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

        for n in tqdm(range(len(mask_ids)), total=len(mask_ids)):
            mask = imread(MASK_PATH + mask_ids[n])
            Y[n] = mask
        
        return X, Y
    else:
        return X

def create_loader(X, Y, batch_size = 64, perc_val = 0.2, perc_test = 0.2):
    # Partimos los datos en Train, Validation, y Test
    n_total = len(X)
    
    n_val = int(perc_val*n_total)
    n_test = int(perc_test*n_total)
    n_train = n_total - n_val - n_test

    s_idxs = np.arange(n_total)
    np.random.shuffle(s_idxs)

    train_idxs = s_idxs[:n_train]
    val_idxs = s_idxs[n_train:n_train+n_val]
    test_idxs = s_idxs[-n_test:]

    train_set = SignalDataset(X[train_idxs], Y[train_idxs])
    validation_set = SignalDataset(X[val_idxs], Y[val_idxs])
    test_set = SignalDataset(X[test_idxs], Y[test_idxs])

    # Generamos dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=0)
    return train_loader, validation_loader, test_loader

def dice(mask1, mask2):
    mask1 = (mask1 > 0.5)
    mask2 = (mask2 > 0.5)
    
    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum())

def jaccard(mask1, mask2):
    mask1 = (mask1 > 0.5)
    mask2 = (mask2 > 0.5)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def test_similarity(model, data, device = "cuda"):
    dices = []
    jaccards = []
    for input, target in data:
        with torch.no_grad():
            output = model(torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device))
            output = output.squeeze().cpu().numpy()
        dices.append(dice(target, output))
        jaccards.append(jaccard(target, output))
    print(f"Dice: {np.mean(dices)}")
    print(f"Jaccard: {np.mean(jaccards)}")
    return np.mean(dices), np.mean(jaccards)