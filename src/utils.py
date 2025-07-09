import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json
import time
from src import models


class SignalDataset(Dataset):
    def __init__(self, X, Y):
        if not torch.is_tensor(X):
            self.X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            self.Y = torch.Tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def select_device(verbose = True):
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        if verbose:
            print("GPU disponible")
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU no disponible, usando CPU")
    
    return device

def load_images(DATA_PATH = "data_y", getMask = True, verbose = True, IMG_CHANNELS = 3):
    pass

def create_loader(data, batch_size = 64):
    X, Y = data
    set = SignalDataset(X, Y)
    loader = DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def save_json(HISTORY_PATH, selected_model, train_losses, validation_losses, lr, n_epochs, patience, dropout_rate):

    JSON_FILE = os.path.join(HISTORY_PATH, f"{selected_model}.json")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE) as file:
            history = json.load(file)  
    else:
        history = {}

    history[len(history)] = {"number of epochs": f"{len(validation_losses)}/{n_epochs}",
                    "patience": patience,
                    "learn rate": lr,
                    "dropout": dropout_rate,
                    "train loss": train_losses,
                    "validation loss" : validation_losses}

    with open(JSON_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def select_model(model_options, configuration, verbose):
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
    
    return selected_model

def train(model, loader, isClasification, get_mask, optimizer, criterion, device):
    total_loss = 0.0
    for data in loader:
        model.train()
        inputs, targets = data
        inputs = inputs.to(device)
        if isClasification:
            targets = targets.squeeze().long()
            targets.to(device)
        else:
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


def test(model, loader, isClasification, get_mask, criterion, device):
    model.eval()
    total_loss = 0.0
    for data in loader:
        inputs, targets = data
        inputs = inputs.to(device)
        if isClasification:
            targets = targets.squeeze().long()
            targets.to(device)
        else:
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

def set_model(selected_model, configuration, BASE_DIR, verbose):
    MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
    IMG_HEIGHT = configuration["image"]["height"]
    IMG_WIDTH = configuration["image"]["width"]
    get_mask = configuration["models"][selected_model]["get mask"]
    model_name = configuration["models"][selected_model]["model"]
    criterion_configuration = configuration["models"][selected_model]["criterion"]
    optimizer_configuration = configuration["models"][selected_model]["optimizer"]
    use_saved_model = configuration["train"]["use saved"]
    dropout_rate = configuration["train"]["dropout"]
    lr = configuration["train"]["learn rate"]
    

    device = select_device(verbose)
    if device == torch.device("cuda"):
        gpu_ram = configuration["train"]["gpu ram"]
        batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH))
    else:
        batch_size = configuration["train"]["batch size"]
    
    if verbose:
        print(f"batch size: {batch_size}")
    
    if use_saved_model:
        use_saved_model = os.path.exists(os.path.join(MODEL_PATH, f"{selected_model}.pth"))

    model = getattr(models, model_name[0])(dropout_rate = dropout_rate)
    if use_saved_model:
        if verbose:
            print("Usando salvado")
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth")))
    else:
        if len(model_name) != 1:
            try:  
                aux = getattr(models, model_name[1])(dropout_rate = dropout_rate)
                aux.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name[1]}.pth")))
                model.encoder.load_state_dict(aux.encoder.state_dict())
            except: "Modelo de transfer learning no ecnontrado"

    if len(model_name) != 1:
        for param in model.encoder.parameters():
            param.requires_grad = False
    model.to(device)

    criterion = getattr(torch.nn, criterion_configuration)()
    optimizer = getattr(torch.optim, optimizer_configuration)(model.parameters(), lr=lr)

    return model, MODEL_PATH, dropout_rate, device, get_mask, lr, criterion, optimizer, batch_size

def set_train(configuration):
    n_epochs = configuration["train"]["epochs"]
    patience = configuration["train"]["patience"]
    print_epoch = configuration["train"]["print epoch"]

    return n_epochs, patience, print_epoch

def train_loop(selected_model, MODEL_PATH, model, optimizer, criterion, train_loader, validation_loader, n_epochs, patience, dropout_rate, lr, device, isClasification, get_mask, HISTORY_PATH, verbose, print_epoch):
    best_val_loss = test(model, validation_loader, isClasification, get_mask, criterion, device)
    best_train_loss = test(model, train_loader, isClasification, get_mask, criterion, device)

    train_losses = []
    validation_losses = []

    j = 0
    epoch = 1
    if verbose:
        print(f"val_loss: {best_val_loss}")
        print(f"train_loss: {best_train_loss}")
        print(f"Train:\n    Patience: {patience}\n    Dropout: {dropout_rate}")
        start = time.time()
        i_time = time.time()
    best_epoch = epoch
    while ((epoch <= n_epochs) and (j < patience)):
        train_loss = train(model, train_loader, isClasification, get_mask, optimizer, criterion, device)
        validation_loss = test(model, validation_loader, isClasification, get_mask, criterion, device)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

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

    model.load_state_dict(selected_model_state_dict)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{selected_model}.pth"))
    save_json(HISTORY_PATH, selected_model, train_losses, validation_losses, lr, n_epochs, patience, dropout_rate)

    if verbose:
        print('La mejor Epoca fue {:03d}, Train loss: {:.4f}, Validation loss: {:.4f}'.format(best_epoch, best_train_loss, best_val_loss))
        end = time.time()
        elapsed = end-start
        print(f"Tiempo de ejecución: {int(elapsed // 3600)} horas, {int((elapsed // 60) % 60)} minutos, {int(elapsed % 60)} segundos")

    return model

def graph(BASE_DIR):
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

    model_options = configuration["models"]["all"]

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