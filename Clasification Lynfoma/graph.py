import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
    configuration = json.load(file)

parts = ["segmentation", "clasification"]

for part in parts:
    model_options = configuration["models"][part]["all"]

    for selected_model in model_options:

        JSON_PATH = configuration["path"]["history"]
        JSON_FILE = os.path.join(BASE_DIR, JSON_PATH, f"{selected_model}.json")

        if os.path.exists(JSON_FILE):

            with open(os.path.join(JSON_FILE)) as file:
                history = json.load(file)
            
            train_loss = history[str(len(history)-1)]["train loss"]
            validation_loss = history[str(len(history)-1)]["validation loss"]

            plt.figure()
            plt.plot(train_loss, label = "train")
            plt.plot(validation_loss, label = "validation")
            plt.xlabel('Epocas')
            plt.ylabel('Loss')
            plt.title(f"Loss {selected_model}")
            plt.legend()

            plt.savefig(os.path.join(BASE_DIR, "hito", f"{selected_model}_loss.png"))

        else:
            print(f"json file {JSON_FILE} not found")