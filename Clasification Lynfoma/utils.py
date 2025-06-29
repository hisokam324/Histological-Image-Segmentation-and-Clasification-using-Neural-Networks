import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import json

class SignalDatasetSeg(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.Y = torch.tensor(Y, dtype=torch.float32) if not torch.is_tensor(Y) else Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SignalDatasetCals(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.Y = torch.tensor(Y, dtype=torch.long)  

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

def load_images(DATA_PATH = "data_y", isClasification = True, verbose = True, IMG_CHANNELS = 3):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)
    
    if isClasification:
        IMG_HEIGHT = configuration["image"]["clasification"]["heigth"]
        IMG_WIDTH = configuration["image"]["clasification"]["width"]
    else: 
        IMG_HEIGHT = configuration["image"]["segmentation"]["heigth"]
        IMG_WIDTH = configuration["image"]["segmentation"]["width"]
    

    IMAGE_PATH = os.path.join(DATA_PATH, "Image")
    
    image_ids = sorted(os.listdir(IMAGE_PATH))
    
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    if verbose:
        print("Images:")
        for n in tqdm(range(len(image_ids)), total=len(image_ids)):
            img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
            X[n] = img
        
        X = X.transpose(0, 3, 1, 2)

        if isClasification:
            print("Classes:")
            classes = pd.read_csv(os.path.join(DATA_PATH, "Estimation.csv"), delimiter=',')
            Y = classes["Class"].tolist()
        else:
            print("Masks:")
            MASK_PATH = os.path.join(DATA_PATH, "Mask")
            mask_ids = sorted(os.listdir(MASK_PATH))
            Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

            for n in tqdm(range(len(mask_ids)), total=len(mask_ids)):
                mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
                Y[n] = mask
            
    else:
        for n in range(len(image_ids)):
            img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
            X[n] = img
        
        X = X.transpose(0, 3, 1, 2)

        if isClasification:
            classes = pd.read_csv(os.path.join(DATA_PATH, "Estimation.csv"), delimiter=',')
            Y = classes["Class"].tolist()
        else:
            MASK_PATH = os.path.join(DATA_PATH, "Mask")
            mask_ids = sorted(os.listdir(MASK_PATH))
            Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

            for n in range(len(mask_ids)):
                mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
                Y[n] = mask
            
    return (X, Y)

def create_loader(data, batch_size = 64, isClasification = True):
    X, Y = data
    if isClasification:
        set = SignalDatasetCals(X, Y)
    else :
        set = SignalDatasetSeg(X, Y)
    loader = DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def dice(mask1, mask2):
    mask1 = (mask1 > 0.5)
    mask2 = (mask2 > 0.5)
    
    intersection = np.logical_and(mask1, mask2).sum()
    return 2 * intersection / (mask1.sum() + mask2.sum())

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

def test_clasification(model, loader, device = "cuda", verbose = True):
    ceros = 0
    ones = 0
    twos = 0
    correct = 0
    total = 0
    model.eval()
    for data in tqdm(loader):
        inputs, targets = data
        with torch.no_grad():
            outputs = model(inputs.to(device))
            outputs = outputs.cpu()
            for i in range(len(outputs)):
                total += 1
                if (torch.argmax(outputs[i]).item() == targets[i].item()):
                    correct += 1
                if (torch.argmax(outputs[i]).item() == 0):
                    ceros += 1
                if (torch.argmax(outputs[i]).item() == 1):
                    ones += 1
                if (torch.argmax(outputs[i]).item() == 2):
                    twos += 1
    return correct/total, ceros/total, ones/total, twos/total

def save_json(selected_model, train_losses, validation_losses, lr, n_epochs, patience, dropout_rate, use_saved_model):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

    JSON_PATH = configuration["path"]["history"]
    JSON_FILE = os.path.join(BASE_DIR, JSON_PATH, f"{selected_model}.json")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE) as file:
            history = json.load(file)  
    else:
        history = {}

    history[len(history)] = {"number of epochs": f"{len(validation_losses)}/{n_epochs}",
                    "patience": patience,
                    "learn rate": lr,
                    "dropout": dropout_rate,
                    "use saved model": use_saved_model,
                    "train loss": train_losses,
                    "validation loss" : validation_losses}

    with open(JSON_FILE, 'w') as file:
        json.dump(history, file, indent=4)