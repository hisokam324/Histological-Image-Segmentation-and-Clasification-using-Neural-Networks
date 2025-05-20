import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import json

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

def select_device(verbose = True):
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        if verbose:
            print("GPU disponible")
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU no disponible, usando CPU")
    
    return device

def load_images(DATA_PATH = "data_y", IMG_CHANNELS = 3, getMask = True, verbose = True):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

    IMG_HEIGHT = configuration["image"]["heigth"]
    IMG_WIDTH = configuration["image"]["width"]

    IMAGE_PATH = os.path.join(DATA_PATH, "Image")
    
    image_ids = sorted(os.listdir(IMAGE_PATH))
    
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    if verbose:
        print("imagenes")
        for n in tqdm(range(len(image_ids)), total=len(image_ids)):
            img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
            X[n] = img
        
        X = X.transpose(0, 3, 1, 2)

        if getMask:
            print("mascaras")
            MASK_PATH = os.path.join(DATA_PATH, "Mask")
            mask_ids = sorted(os.listdir(MASK_PATH))
            Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

            for n in tqdm(range(len(mask_ids)), total=len(mask_ids)):
                mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
                Y[n] = mask
            
            return (X, Y)
        else:
            return (X, X)
    else:
        for n in range(len(image_ids)):
            img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
            X[n] = img
        
        X = X.transpose(0, 3, 1, 2)

        if getMask:
            MASK_PATH = os.path.join(DATA_PATH, "Mask")
            mask_ids = sorted(os.listdir(MASK_PATH))
            Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

            for n in range(len(mask_ids)):
                mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
                Y[n] = mask
            
            return (X, Y)
        else:
            return (X, X)

def create_loader(data, batch_size = 64):
    X, Y = data
    set = SignalDataset(X, Y)
    loader = DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

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

def test_similarity(model, loader, device = "cuda", verbose = True):
    dices = []
    jaccards = []
    model.eval()
    for data in tqdm(loader):
        inputs, targets = data
        with torch.no_grad():
            outputs = model(inputs.to(device))
            outputs = outputs.squeeze().cpu().numpy()
        dices.append(dice(targets.squeeze().numpy(), outputs))
        jaccards.append(jaccard(targets.squeeze().numpy(), outputs))
    if verbose:
        print(f"Dice: {np.mean(dices)}")
        print(f"Jaccard: {np.mean(jaccards)}")
    return np.mean(dices), np.mean(jaccards)