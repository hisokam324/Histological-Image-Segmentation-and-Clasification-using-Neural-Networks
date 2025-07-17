import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.io import imread
from src import utils

def load_images(SPLIT_PATH, configuration, IMG_CHANNELS = 3):
    verbose = configuration["train"]["verbose"]
    isClasification = configuration["train"]["is clasification"]
    IMG_HEIGHT = configuration["image"]["height"]
    IMG_WIDTH = configuration["image"]["width"]
    IMAGE_PATH = os.path.join(SPLIT_PATH, "Image")
    
    image_ids = sorted(os.listdir(IMAGE_PATH))
    
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    if verbose:
        print("Images:")
    for n in tqdm(range(len(image_ids)), total=len(image_ids), disable=not verbose):
        img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
        X[n] = img
    
    X = X.transpose(0, 3, 1, 2)

    if isClasification:
        classes = pd.read_csv(os.path.join(SPLIT_PATH, "Estimation.csv"), delimiter=',')
        Y = classes["Class"].tolist()
    else:
        if verbose:
            print("Masks:")
        MASK_PATH = os.path.join(SPLIT_PATH, "Mask")
        mask_ids = sorted(os.listdir(MASK_PATH))
        Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

        for n in tqdm(range(len(mask_ids)), total=len(mask_ids), disable=not verbose):
            mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
            Y[n] = mask
            
    return (X, Y)

def get_loaders(configuration, toLoad):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, configuration["path"]["data"])
    data_division = configuration["path"]["data division"]
    batch_size = configuration["train"]["batch size"]

    loaders = []

    for i in range(len(toLoad)):
        if toLoad[i]:
            loaders.append(utils.create_loader(load_images(os.path.join(DATA_PATH, data_division[i]), configuration), batch_size=batch_size))
        else:
            loaders.append([])

    return loaders[0], loaders[1], loaders[2]