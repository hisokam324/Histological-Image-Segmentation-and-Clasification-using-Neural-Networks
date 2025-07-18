"""
Auxiliary module to adapt original data
"""

import cv2
import json
import os
import pandas as pd
from tqdm import tqdm

def save(img, PATH, name, j, extension, rotations):
    """
    Auxiliary function to save an image and its rotations

    Args:
        img (Numpy Array): Image to save

        PATH (String): Image base path

        name (String): Image extention path

        j (Intager): Image number, to create image name

        extension (String): Image extention, png or bmp

        rotations (Intager): Number of rotations to save, 2 or 4
    """
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
    """
    Auxiliary function to crop images

    Args:
        img (Numpy Array): Image to crop

        division_h (Intager): height division

        division_w (Intager): width division

        mask (Boolean): Indicates if image has only one channel
    
        Returns:
            out (List[Numpy Array]): List of cropped images
    """
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

def resize(img, factor):
    """
    Auxiliary function to resize images. Output size has module 16

    Args:
        img (Numpy Array): Image to resize

        factor (Tuple[Intager, Intager]): Height and width factor to resize image

    Returns:
        out (Numpy Array): Resized image
    """
    if factor != 1:
        h, w = img.shape[:2]
        hh, ww = h//factor, w//factor
        out = cv2.resize(img, (ww, hh))
    else:
        out = img
    h, w = out.shape[:2]
    hh, ww = h//16, w//16
    return out[:hh*16, :ww*16]
                  
def cutExtention(path, cut_by = ".tiff"):
    """
    Auxiliary function to cut extension of image path

    Args:
        path (String): Image path

        cut_by (String): Flag to cut by

    Returns:
        base_path (String): Path without extention

        extention_path (String): Extention of path
    """
    idx = path.find(cut_by)
    return path[:idx], path[idx:]

def main():
    """
    Run code:
        Clasification:
            Load original data

            Resize images

            Save images

        Segmentation:
            Load original data

            Crop images

            Save images
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configurationAdapter.json')) as file:
            configuration = json.load(file)


    CLASIFICATION_CLASSES = configuration["adapt dataset"]["clasification"]["classes"]
    CLASIFICATION_FACTOR = configuration["adapt dataset"]["clasification"]["factor"]
    CLASIFICATION_ROTATIONS = configuration["adapt dataset"]["clasification"]["rotations"]
    SEGMENTATION_DIVISION_H = configuration["adapt dataset"]["segmentation"]["division h"]
    SEGMENTATION_DIVISION_W = configuration["adapt dataset"]["segmentation"]["division w"]
    SEGMENTATION_ROTATIONS = configuration["adapt dataset"]["segmentation"]["rotations"]
    VALIDATION_FRACTION = configuration["adapt dataset"]["validation"]
    TEST_FRACTION = configuration["adapt dataset"]["test"]

    GENERAL_PATH = os.path.join(BASE_DIR, configuration["path"]["general data"])
    SEGMENTATION_PATH = os.path.join(GENERAL_PATH, configuration["path"]["segmentation data"])
    CLASIFICATION_PATH_TRAIN = os.path.join(GENERAL_PATH, configuration["path"]["clasification data"][0])
    CLASIFICATION_PATH_TEST1 = os.path.join(GENERAL_PATH, configuration["path"]["clasification data"][1])
    CLASIFICATION_PATH_TEST2 = os.path.join(GENERAL_PATH, configuration["path"]["clasification data"][2])

    DATASET_CLASIFICATION = os.path.join(BASE_DIR, configuration["path"]["dataset clasification"])
    DATASET_SEGMENTATION = os.path.join(BASE_DIR, configuration["path"]["dataset segmentation"])
    DATA_DIVISION = configuration["path"]["data division"]

    c_images = os.listdir(os.path.join(CLASIFICATION_PATH_TRAIN, "Images"))
    c_class = pd.read_csv(os.path.join(CLASIFICATION_PATH_TRAIN, "Estimation.csv"), delimiter=';', index_col="File")
    s_images = os.listdir(os.path.join(SEGMENTATION_PATH, "png"))

    min_class = min(c_class['Class'].value_counts().sort_index())
    validation_split = int(max(min_class*VALIDATION_FRACTION, 1))
    train_split = min_class-validation_split


    print("CLASIFICACION:")

    iter_train = 0
    iter_validation = 0
    clasification_train = []
    clasification_validation = []
    for i in tqdm(range(train_split)):
        for class_x in range(CLASIFICATION_CLASSES):
            image = c_class[c_class["Class"] == class_x].index[i]
            img = cv2.imread(os.path.join(CLASIFICATION_PATH_TRAIN, "Images", image))
            img = resize(img, CLASIFICATION_FACTOR)
            save(img, os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[0]), "Image", iter_train, "png", CLASIFICATION_ROTATIONS)
            for _ in range(CLASIFICATION_ROTATIONS):
                clasification_train.append(class_x)
            iter_train += 1

    for i in tqdm(range(validation_split)):
        for class_x in range(CLASIFICATION_CLASSES):
            image = c_class[c_class["Class"] == class_x].index[i+train_split]
            img = cv2.imread(os.path.join(CLASIFICATION_PATH_TRAIN, "Images", image))
            img = resize(img, CLASIFICATION_FACTOR)
            save(img, os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[1]), "Image", iter_validation, "png", CLASIFICATION_ROTATIONS)
            for _ in range(CLASIFICATION_ROTATIONS):
                clasification_validation.append(class_x)
            iter_validation += 1

    h, w = img.shape[:2]
    with open(os.path.join(BASE_DIR, 'configurationClasification.json')) as file:
            configurationClasification = json.load(file)

    configurationClasification["image"]["heigth"] = h
    configurationClasification["image"]["width"] = w
    configurationClasification["path"]["data"] = configuration["path"]["dataset clasification"]
    configurationClasification["path"]["data division"] = configuration["path"]["data division"]

    with open(os.path.join(BASE_DIR, 'configurationClasification.json'), 'w') as file:
        json.dump(configurationClasification, file, indent=4)

    clasification_train = pd.DataFrame(clasification_train, columns=["Class"])
    clasification_train.index.name = "Image"
    clasification_train.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[0], "Estimation.csv"))
    clasification_validation = pd.DataFrame(clasification_validation, columns=["Class"])
    clasification_validation.index.name = "Image"
    clasification_validation.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[1], "Estimation.csv"))

    # Test canino

    c_images = os.listdir(os.path.join(CLASIFICATION_PATH_TEST1))
    aux = []
    for i in c_images:
        if (i.find(".tiff") != -1):
            aux.append(i)
    c_images = aux
    c_class = pd.read_csv(os.path.join(CLASIFICATION_PATH_TEST1, "Estimation.csv"), delimiter=';', index_col="File")

    iter_test = 0
    clasification_test = []
    for image in tqdm(c_images):
        clasification = c_class.loc[image]['Class']
        img = cv2.imread(os.path.join(CLASIFICATION_PATH_TEST1, image))
        img = resize(img, CLASIFICATION_FACTOR)
        save(img, os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[2]), "Image", iter_test, "png", CLASIFICATION_ROTATIONS)
        for _ in range(CLASIFICATION_ROTATIONS):
            clasification_test.append(clasification)
        iter_test += 1

    clasification_test = pd.DataFrame(clasification_test, columns=["Class"])
    clasification_test.index.name = "Image"
    clasification_test.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[2], "Estimation.csv"))

    # Test Felino

    c_images = os.listdir(os.path.join(CLASIFICATION_PATH_TEST2))
    aux = []
    for i in c_images:
        if (i.find(".tiff") != -1):
            aux.append(i)
    c_images = aux
    c_class = pd.read_csv(os.path.join(CLASIFICATION_PATH_TEST2, "Estimation.csv"), delimiter=';', index_col="File")

    iter_test = 0
    clasification_test = []
    for image in tqdm(c_images):
        clasification = c_class.loc[image]['Class']
        img = cv2.imread(os.path.join(CLASIFICATION_PATH_TEST2, image))
        img = resize(img, CLASIFICATION_FACTOR)
        save(img, os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[3]), "Image", iter_test, "png", CLASIFICATION_ROTATIONS)
        for _ in range(CLASIFICATION_ROTATIONS):
            clasification_test.append(clasification)
        iter_test += 1

    clasification_test = pd.DataFrame(clasification_test, columns=["Class"])
    clasification_test.index.name = "Image"
    clasification_test.to_csv(os.path.join(DATASET_CLASIFICATION, DATA_DIVISION[3], "Estimation.csv"))

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

    h, w = cropped_img[0].shape[:2]
    with open(os.path.join(BASE_DIR, 'configurationSegmentation.json')) as file:
            configurationSegmentation = json.load(file)

    configurationSegmentation["image"]["heigth"] = h
    configurationSegmentation["image"]["width"] = w
    configurationSegmentation["path"]["data"] = configuration["path"]["dataset segmentation"]
    configurationSegmentation["path"]["data division"] = configuration["path"]["data division"][:2]

    with open(os.path.join(BASE_DIR, 'configurationSegmentation.json'), 'w') as file:
        json.dump(configurationSegmentation, file, indent=4)


if __name__ == "__main__":
    main()