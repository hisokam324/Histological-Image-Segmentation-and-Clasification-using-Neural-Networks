import cv2
import json
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def save(img, PATH, name, j, extension, rotations):
    img1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img, cv2.ROTATE_180)
    img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rotations == 2:
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)}.{extension}"), img)
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)+1}.{extension}"), img2)
    if rotations == 4:
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)}.{extension}"), img)
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)+1}.{extension}"), img1)
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)+2}.{extension}"), img2)
        cv2.imwrite(os.path.join(PATH, name, f"{(j*rotations)+3}.{extension}"), img3)

def crop(img, division_h, division_w, mask = False):
    out = []
    h, w = img.shape[:2]
    hh, ww = h//division_h, w//division_w
    for i in range(division_h):
        for j in range(division_w):
            if mask:
                out.append(img[hh*i:hh*(i+1), ww*j:ww*(j+1)])
            else:
                out.append(img[hh*i:hh*(i+1), ww*j:ww*(j+1), :])
    return out
                  
def cutExtention(path, cut_by = ".tiff"):
      idx = path.find(cut_by)
      return path[:idx], path[idx:]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

CLASIFICATION_ROTATIONS = configuration["adapt dataset"]["clasification rotations"]
SEGMENTATION_ROTATIONS = configuration["adapt dataset"]["segmentation rotations"]
CLASIFICATION_DIVISION_H = configuration["adapt dataset"]["clasification division h"]
CLASIFICATION_DIVISION_W = configuration["adapt dataset"]["clasification division w"]
SEGMENTATION_DIVISION_H = configuration["adapt dataset"]["segmentation division h"]
SEGMENTATION_DIVISION_W = configuration["adapt dataset"]["segmentation division w"]
VALIDATION_FRACTION = configuration["adapt dataset"]["validation"]
TEST_FRACTION = configuration["adapt dataset"]["test"]

SPLISTS = CLASIFICATION_DIVISION_H*CLASIFICATION_DIVISION_W
VALIDATION_SPLIT = int(VALIDATION_FRACTION*(SPLISTS)//1+1)
TEST_SPLIT = int(TEST_FRACTION*(SPLISTS)//1+1)
TRAIN_SPLIT = SPLISTS-VALIDATION_SPLIT-TEST_SPLIT

GENERAL_PATH = os.path.join(BASE_DIR, configuration["path"]["general data"])
SEGMENTATION_PATH = os.path.join(GENERAL_PATH, configuration["path"]["segmentation data"])
CLASIFICATION_PATH = os.path.join(GENERAL_PATH, configuration["path"]["clasification data"])

DATASET_CLASIFICATION = os.path.join(BASE_DIR, configuration["path"]["dataset clasification"])
DATASET_SEGMENTATION = os.path.join(BASE_DIR, configuration["path"]["dataset segmentation"])
DATA_DIVISION = configuration["path"]["data division"]

c_images = os.listdir(os.path.join(CLASIFICATION_PATH, "Images"))
c_class = pd.read_csv(os.path.join(CLASIFICATION_PATH, "Estimation.csv"), delimiter=';', index_col="File")
s_images = os.listdir(os.path.join(SEGMENTATION_PATH, "png"))

print("CLASIFICACION:")

iter_train = 0
iter_validation = 0
iter_test = 0
clasification_train = []
clasification_validation = []
clasification_test = []
for image in tqdm(c_images):
    class_train = []
    clasification = c_class.loc[image]['Class']
    img = cv2.imread(os.path.join(CLASIFICATION_PATH, "Images", image))
    cropped = crop(img, CLASIFICATION_DIVISION_H, CLASIFICATION_DIVISION_W)
    for i in range(TRAIN_SPLIT):
        save(cropped[i], os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[0]), "Image", iter_train, "png", CLASIFICATION_ROTATIONS)
        for j in range(CLASIFICATION_ROTATIONS):
            clasification_train.append(clasification)
        iter_train += 1
    for i in range(VALIDATION_SPLIT):
        save(cropped[i+TRAIN_SPLIT], os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[1]), "Image", iter_validation, "png", CLASIFICATION_ROTATIONS)
        for j in range(CLASIFICATION_ROTATIONS):
            clasification_validation.append(clasification)
        iter_validation += 1
    for i in range(TEST_SPLIT):
        save(cropped[i+TRAIN_SPLIT+VALIDATION_SPLIT], os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[2]), "Image", iter_test, "png", CLASIFICATION_ROTATIONS)
        for j in range(CLASIFICATION_ROTATIONS):
            clasification_test.append(clasification)
        iter_test += 1

clasification_train = pd.DataFrame(clasification_train, columns=["Class"])
clasification_train.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[0], "Estimation.csv"))
clasification_validation = pd.DataFrame(clasification_validation, columns=["Class"])
clasification_validation.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[1], "Estimation.csv"))
clasification_test = pd.DataFrame(clasification_test, columns=["Class"])
clasification_test.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[2], "Estimation.csv"))

print("SEGMENTATION:")

aux = []
for image in s_images:
    name, extention = cutExtention(image)
    if name not in aux:
        aux.append(name)

s_images = aux

segmentation_total = len(s_images)
validation_split = int(VALIDATION_FRACTION*segmentation_total)
test_split = int(TEST_FRACTION*segmentation_total)
train_split = segmentation_total-validation_split-test_split

segmentation_iter = 0
iter_train = 0
iter_validation = 0
iter_test = 0
for image in tqdm(s_images):
    img  = cv2.imread(os.path.join(SEGMENTATION_PATH, "png", f"{image}.tiff.png"))
    mask  = cv2.imread(os.path.join(SEGMENTATION_PATH, "png", f"{image}.tiff_label.png"))[:,:,1]
    mask[mask > 0] = 255
    cropped_img = crop(img, SEGMENTATION_DIVISION_H, SEGMENTATION_DIVISION_W)
    cropped_mask = crop(mask, SEGMENTATION_DIVISION_H, SEGMENTATION_DIVISION_W, True)
    if segmentation_iter < train_split:
        for i in range(len(cropped_img)):
            save(cropped_img[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[0]), "Image", iter_train, "png", SEGMENTATION_ROTATIONS)
            save(cropped_mask[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[0]), "Mask", iter_train, "bmp", SEGMENTATION_ROTATIONS)
            iter_train += 1
    elif segmentation_iter < train_split+validation_split:
        for i in range(len(cropped_img)):
            save(cropped_img[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[1]), "Image", iter_validation, "png", SEGMENTATION_ROTATIONS)
            save(cropped_mask[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[1]), "Mask", iter_validation, "bmp", SEGMENTATION_ROTATIONS)
            iter_validation += 1
    else:
        for i in range(len(cropped_img)):
            save(cropped_img[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[2]), "Image", iter_test, "png", SEGMENTATION_ROTATIONS)
            save(cropped_mask[i], os.path.join(DATASET_SEGMENTATION, DATA_DIVISION[2]), "Mask", iter_test, "bmp", SEGMENTATION_ROTATIONS)
            iter_test += 1

    segmentation_iter += 1
     


