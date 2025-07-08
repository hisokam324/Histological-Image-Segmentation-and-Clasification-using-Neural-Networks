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
model_options = configuration["models"]["all"]

if verbose:
    selecting_model = True
    while selecting_model:
        try:
            print("Selec model:")
            for i in range(len(model_options)):
                print(f"{i} {model_options[i]}")
            selected = int(input("modelo: "))
            selected = selected%len(model_options)
            selecting_model = False
        except:
            print("Modelo invalido, por favor ingrede un numero (integer)")
    selected_model = model_options[selected]
    print(f"modelo seleccionado: {selected} {selected_model}")
else :
    selected = configuration["train"]["select model"]
    selected_model = model_options[selected]

MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
IMG_HEIGHT = configuration["image"]["heigth"]
IMG_WIDTH = configuration["image"]["width"]
model_name = configuration["models"][selected_model]["model"]
criterion_configuration = configuration["models"][selected_model]["criterion"]
optimizer_configuration = configuration["models"][selected_model]["optimizer"]
use_saved_model = configuration["train"]["use saved"]
dropout_rate = configuration["train"]["dropout"]
lr = configuration["train"]["learn rate"]
n_epochs = configuration["train"]["epochs"]
patience = configuration["train"]["patience"]
print_epoch = configuration["train"]["print epoch"]
n_classes = configuration["train"]["classes"]

device = utils.select_device(verbose)
if device == torch.device("cuda"):
    gpu_ram = configuration["train"]["gpu ram"]
    batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH*2))
else:
    batch_size = configuration["train"]["batch size"]


model = getattr(models, model_name[0])(dropout_rate = dropout_rate, out_classes=n_classes, img_heigth=IMG_HEIGHT, img_width=IMG_WIDTH)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth")))
model.to(device)

train_loader, validation_loader, test_loader = utils.create_loader(batch_size=batch_size)

print(f"Test Train: {utils.test_clasification(model, train_loader)}")
print(f"Test Validation: {utils.test_clasification(model, validation_loader)}")
print(f"Test Test: {utils.test_clasification(model, test_loader)}")