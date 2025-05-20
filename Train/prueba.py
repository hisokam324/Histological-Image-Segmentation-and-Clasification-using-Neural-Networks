import torch
import json
import os
import time
from tqdm import tqdm
import utils
import models

def train(model,loader, get_mask):
    total_loss = 0.0
    for data in loader:
        model.train()
        inputs, targets = data
        inputs = inputs.to(device)
        if get_mask:
            targets = targets.to(device).unsqueeze(1)
        else:
            targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    total_loss/= len(loader.dataset)
    return total_loss 


def test(model,loader, get_mask):
    model.eval()
    total_loss = 0.0
    for data in loader:
        inputs, targets = data
        inputs = inputs.to(device)
        if get_mask:
            targets = targets.to(device).unsqueeze(1)
        else:
            targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    total_loss/= len(loader.dataset)
    return total_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

verbose = configuration["train"]["verbose"]
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
DATA_PATH = os.path.join(BASE_DIR,configuration["path"]["data"])
data_division = configuration["path"]["data division"]
IMG_HEIGHT = configuration["image"]["heigth"]
IMG_WIDTH = configuration["image"]["width"]
IN_CHANNELS = configuration["models"][selected_model]["in channels"]
OUT_CHANNELS = configuration["models"][selected_model]["out channels"]
get_mask = configuration["models"][selected_model]["get mask"]
name = configuration["models"][selected_model]["name"]
criterion_configuration = configuration["models"][selected_model]["criterion"]
optimizer_configuration = configuration["models"][selected_model]["optimizer"]
save_model = configuration["models"][selected_model]["save"]
use_saved_model = configuration["train"]["use saved"]
dropout_rate = configuration["train"]["dropout"]
lr = configuration["train"]["learn rate"]
n_epochs = configuration["train"]["epochs"]
patience = configuration["train"]["patience"]
print_epoch = configuration["train"]["print epoch"]


device = utils.select_device(verbose)
if device == torch.device("cuda"):
    gpu_ram = configuration["train"]["gpu ram"]
    batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH))
else:
    batch_size = configuration["train"]["batch size"]

if verbose:
    print(f"batch size: {batch_size}")

train_loader = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[0]), IN_CHANNELS, get_mask, verbose = verbose), batch_size=batch_size)
validation_loader = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[1]), IN_CHANNELS, get_mask, verbose = verbose), batch_size=batch_size)
test_loader = utils.create_loader(utils.load_images(os.path.join(DATA_PATH, data_division[2]), IN_CHANNELS, get_mask, verbose = verbose), batch_size=batch_size)

model = getattr(models, name)(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, dropout_rate= dropout_rate)
if use_saved_model:
    print("Usando salvado")
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{save_model}.pth")))
model.to(device)

criterion = getattr(torch.nn, criterion_configuration)()
optimizer = getattr(torch.optim, optimizer_configuration)(model.parameters(), lr=lr)

selected_model_state_dict = model.state_dict()

best_val_loss = test(model, validation_loader, get_mask)
best_train_loss = test(model, train_loader, get_mask)
j = 0
epoch = 0
if verbose:
    print(f"val_loss: {best_val_loss}")
    print(f"train_loss: {best_train_loss}")
    print(f"Train:\n    Patience: {patience}\n    Dropout: {dropout_rate}")
    start = time.time()
    i_time = time.time()
best_epoch = epoch
while ((epoch < n_epochs) and (j < patience)):
    train_loss = train(model, train_loader, get_mask)
    validation_loss = test(model, validation_loader, get_mask)

    if epoch%print_epoch == 0 and verbose:
        i_elapsed = time.time()-i_time
        
        print('Epoca: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Train loss: {:.4f}".format(train_loss), ", Validation loss: {:.4f}".format(validation_loss), ", best: {:.4f}".format(min(best_val_loss, validation_loss)), f", Elapsed: {int((i_elapsed // 60) % 60)} minutos, {int(i_elapsed % 60)} segundos")
        i_time = time.time()
        
    if (validation_loss <= best_val_loss):
        best_epoch = epoch
        best_train_loss = train_loss
        best_val_loss = validation_loss
        selected_model_state_dict = model.state_dict()
        j = 0
    else:
        j += 1

    epoch += 1

# Nos quedamos con el modelo con mejor valor de validación 
model.load_state_dict(selected_model_state_dict)

if verbose:
    print('La mejor Epoca fue {:03d}, Train loss: {:.4f}, Validation loss: {:.4f}'.format(
              best_epoch, best_train_loss, best_val_loss))
    end = time.time()
    elapsed = end-start
    print(f"Tiempo de ejecución: {int(elapsed // 3600)} horas, {int((elapsed // 60) % 60)} minutos, {int(elapsed % 60)} segundos")

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{save_model}.pth"))

    if get_mask:
        print("train:")
        utils.test_similarity(model, train_loader, device)
        print("validation:")
        utils.test_similarity(model, validation_loader, device)
        print("test:")
        utils.test_similarity(model, test_loader, device)