import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread
from src import utils

def load_images(DATA_PATH, configuration, getMask = True, verbose = True, IMG_CHANNELS = 3):

    IMG_HEIGHT = configuration["image"]["height"]
    IMG_WIDTH = configuration["image"]["width"]

    IMAGE_PATH = os.path.join(DATA_PATH, "Image")
    
    image_ids = sorted(os.listdir(IMAGE_PATH))
    
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    if verbose:
        print("imagenes")
    for n in tqdm(range(len(image_ids)), total=len(image_ids), disable= not verbose):
        img = imread(os.path.join(IMAGE_PATH, image_ids[n]))
        X[n] = img
    
    X = X.transpose(0, 3, 1, 2)

    if getMask:
        if verbose:
            print("mascaras")
        MASK_PATH = os.path.join(DATA_PATH, "Mask")
        mask_ids = sorted(os.listdir(MASK_PATH))
        Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH), dtype=bool)

        for n in tqdm(range(len(mask_ids)), total=len(mask_ids), disable=not verbose):
            mask = imread(os.path.join(MASK_PATH, mask_ids[n]))
            Y[n] = mask
        
        return (X, Y)
    else:
        return (X, X)
    

def get_loaders(BASE_DIR, configuration, selected_model, toLoad = [True, True, False]):
    DATA_PATH = os.path.join(BASE_DIR,configuration["path"]["data"])
    data_division = configuration["path"]["data division"]
    batch_size = configuration["train"]["batch size"]
    verbose = configuration["train"]["verbose"]
    get_mask = configuration["models"][selected_model]["get mask"]

    loaders = []

    for i in range(len(toLoad)):
        if toLoad[i]:
            loaders.append(utils.create_loader(load_images(os.path.join(DATA_PATH, data_division[i]), configuration, get_mask, verbose = verbose), batch_size=batch_size))
        else:
            loaders.append([])

    return loaders[0], loaders[1], loaders[2]