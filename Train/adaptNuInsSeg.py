import cv2
import json
import os
from tqdm import tqdm

def save(img_path, i, PATH, name, j, extension):
    img = cv2.imread(os.path.join(img_path, os.listdir(img_path)[i]))
    if img.shape != (IMG_HEIGHT, IMG_WIDTH):
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    if extension == "bmp":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img, cv2.ROTATE_180)
    img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imwrite(os.path.join(PATH, name, f"{j}.{extension}"), img)
    cv2.imwrite(os.path.join(PATH, name, f"{j+1}.{extension}"), img1)
    cv2.imwrite(os.path.join(PATH, name, f"{j+2}.{extension}"), img2)
    cv2.imwrite(os.path.join(PATH, name, f"{j+3}.{extension}"), img3)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

IMG_HEIGHT = configuration["image"]["heigth"]
IMG_WIDTH = configuration["image"]["width"]

PATH = os.path.join(BASE_DIR, "archive")
TRAIN_PATH = os.path.join(BASE_DIR, "dataNuInsSeg/train/")
VAL_PATH = os.path.join(BASE_DIR, "dataNuInsSeg/validation/")
TEST_PATH = os.path.join(BASE_DIR, "dataNuInsSeg/test/")

ALTER_PATH = os.path.join(BASE_DIR, "dataNuInsSeg_alter")

folders = os.listdir(PATH)

perc_val = 0.1
perc_test = 0.1

train_iter = 0
val_iter = 0
test_iter = 0
for folder in tqdm(folders):
    img_path = os.path.join(PATH, folder, "tissue images")
    mask_path = os.path.join(PATH, folder, "mask binary")
    mask_alter_path = os.path.join(PATH, folder, "mask binary without border")

    n_total = len(os.listdir(img_path))
    n_val = max(int(perc_val*n_total), 1)
    n_test = max(int(perc_test*n_total), 1)
    n_train = n_total - n_val - n_test

    print(f"train: {n_train}, val: {n_val}, test: {n_test}")

    for i in range(n_train):
        aux = train_iter+i
        save(img_path, i, TRAIN_PATH, "Image", aux*4, "png")
        save(mask_path, i, TRAIN_PATH, "Mask", aux*4, "bmp")
        save(mask_alter_path, i, TRAIN_PATH, "Mask alter", aux*4, "bmp")
    train_iter += n_train
    
    for i in range(n_val):
        aux = val_iter+i
        save(img_path, i, VAL_PATH, "Image", aux*4, "png")
        save(mask_path, i, VAL_PATH, "Mask", aux*4, "bmp")
        save(mask_alter_path, i, VAL_PATH, "Mask alter", aux*4, "bmp")
    val_iter += n_val

    for i in range(n_test):
        aux = test_iter+i
        save(img_path, i, TEST_PATH, "Image", aux*4, "png")
        save(mask_path, i, TEST_PATH, "Mask", aux*4, "bmp")
        save(mask_alter_path, i, TEST_PATH, "Mask alter", aux*4, "bmp")
    test_iter += n_test

alter_iter = 0
for folder in tqdm(folders):
    img_path = os.path.join(PATH, folder, "tissue images")
    mask_path = os.path.join(PATH, folder, "mask binary")
    mask_alter_path = os.path.join(PATH, folder, "mask binary without border")

    n_total = len(os.listdir(img_path))

    for i in range(n_total):
        aux = alter_iter+i
        save(img_path, i, ALTER_PATH, "Image", aux*4, "png")
        save(mask_path, i, ALTER_PATH, "Mask", aux*4, "bmp")
        save(mask_alter_path, i, ALTER_PATH, "Mask alter", aux*4, "bmp")
    alter_iter += n_total