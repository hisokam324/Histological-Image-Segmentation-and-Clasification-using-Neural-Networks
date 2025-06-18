import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import utils
import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

verbose = True
MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
DATA_PATH = os.path.join(BASE_DIR,configuration["path"]["data"])
data_division = configuration["path"]["data division"]
IMG_HEIGHT = configuration["image"]["heigth"]
IMG_WIDTH = configuration["image"]["width"]
dropout_rate = configuration["train"]["dropout"]

device = utils.select_device(verbose)
if device == torch.device("cuda"):
    gpu_ram = configuration["train"]["gpu ram"]
    batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH))
else:
    batch_size = configuration["train"]["batch size"]

loader = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[2]), True))

JSON_FILE = os.path.join(BASE_DIR, "hito", "testAll.json")
results = {}

model_options = configuration["models"]["all"]
for selected_model in model_options:
    get_mask = configuration["models"][selected_model]["get mask"]
    if get_mask:
        model_name = configuration["models"][selected_model]["model"]
        criterion_configuration = configuration["models"][selected_model]["criterion"]
        optimizer_configuration = configuration["models"][selected_model]["optimizer"]
    
        model = getattr(models, model_name[0])(dropout_rate= dropout_rate)

        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth")))
        model.to(device)

        dice, jacard = utils.test_similarity(model, loader)
        results[selected_model] = {"dice": dice, "jacard": jacard}

with open(JSON_FILE, 'w') as file:
    json.dump(results, file, indent=4,)
