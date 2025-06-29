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
model_options = configuration["models"]["clasification"]["all"]

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
DATA_PATH = os.path.join(BASE_DIR,configuration["path"]["dataset clasification"])
data_division = configuration["path"]["data division"]
IMG_HEIGHT = configuration["image"]["clasification"]["heigth"]
IMG_WIDTH = configuration["image"]["clasification"]["width"]
model_name = configuration["models"]["clasification"][selected_model]["model"]
criterion_configuration = configuration["models"]["clasification"][selected_model]["criterion"]
optimizer_configuration = configuration["models"]["clasification"][selected_model]["optimizer"]
use_saved_model = configuration["train"]["use saved"]
dropout_rate = configuration["train"]["dropout"]
lr = configuration["train"]["learn rate"]
n_epochs = configuration["train"]["epochs"]
patience = configuration["train"]["patience"]
print_epoch = configuration["train"]["print epoch"]

device = utils.select_device(verbose)
if device == torch.device("cuda"):
    gpu_ram = configuration["train"]["gpu ram"]
    batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH*2))
else:
    batch_size = configuration["train"]["batch size"]


model = getattr(models, model_name[0])(dropout_rate= dropout_rate)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth")))
model.to(device)

test = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[0]), isClasification = True, verbose = verbose), batch_size = batch_size, isClasification = True)
print(f"Test Canine: {utils.test_clasification(model, test)}")
test = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[1]), isClasification = True, verbose = verbose), batch_size = batch_size, isClasification = True)
print(f"Test Canine: {utils.test_clasification(model, test)}")
test = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[2]), isClasification = True, verbose = verbose), batch_size = batch_size, isClasification = True)
print(f"Test Canine: {utils.test_clasification(model, test)}")
test = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[3]), isClasification = True, verbose = verbose), batch_size = batch_size, isClasification = True)
print(f"Test Feline: {utils.test_clasification(model, test)}")