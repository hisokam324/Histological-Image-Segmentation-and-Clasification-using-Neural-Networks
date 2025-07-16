"""
Se implementan funciones basicas
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm
from src import models


class SignalDataset(Dataset):
    """
    Se implementan dos funciones basicas
    """
    def __init__(self, X, Y):
        """
        Se implementan dos funciones basicas
        """
        if not torch.is_tensor(X):
            self.X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            self.Y = torch.Tensor(Y)

    def __len__(self):
        """
        Se implementan dos funciones basicas
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Se implementan dos funciones basicas
        """
        return self.X[idx], self.Y[idx]

def select_device():
    """
    Funcion auxiliar que identifica la presencia de grafica para realizar los calculos
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = "cuda"
    else:
        device = "cpu"
    
    return device

def create_loader(data, batch_size = 64):
    """
    Funcion auxiliar que crea un loader a partir de data con formato (input, target). Los loaders se crean con un batch size especificado
    """
    X, Y = data
    set = SignalDataset(X, Y)
    loader = DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def save_json(BASE_DIR, configuration, selected_model, train_losses, validation_losses):
    """
    Funcion que salva los resultados obteidos del entrenamiento en un json
    """
    # Carga de datos importantes
    HISTORY_PATH = os.path.join(BASE_DIR, configuration["path"]["history"])
    lr = configuration["train"]["learn rate"]
    n_epochs = configuration["train"]["epochs"]
    patience = configuration["train"]["patience"]
    dropout_rate = configuration["train"]["dropout"]
    # Encontrar Json de modelo
    JSON_FILE = os.path.join(HISTORY_PATH, f"{selected_model}.json")
    # Checkear existencia
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE) as file:
            history = json.load(file)  
    else:
        history = {}

    # Agregar informacion
    history[len(history)] = {"number of epochs": f"{len(validation_losses)}/{n_epochs}",
                    "patience": patience,
                    "learn rate": lr,
                    "dropout": dropout_rate,
                    "train loss": train_losses,
                    "validation loss" : validation_losses}
    #Guardar
    with open(JSON_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def select_model(configuration):
    """
    Funcion auxiliar para seleccionar el modelo a utilizar
    """
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
    
    return selected_model

def train(model, loader, isClasification, get_mask, optimizer, criterion, device):
    """
    Funcion de entrenamiento basica
    (Considera los posibles tipos de entrenamientos esperados, reconstruccion, segmentacion y clasificacion)
    """
    total_loss = 0.0
    for data in loader:
        model.train()
        inputs, targets = data
        inputs = inputs.to(device)
        if isClasification:
            targets = targets.squeeze().long()
        else:
            if get_mask:
                targets = targets.unsqueeze(1)
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
    """
    Funcion de testeo durante entrenamiento basica
    (Considera los posibles tipos de entrenamientos esperados, reconstruccion, segmentacion y clasificacion)
    """
    model.eval()
    total_loss = 0.0
    for data in loader:
        inputs, targets = data
        inputs = inputs.to(device)
        if isClasification:
            targets = targets.squeeze().long()
            
        else:
            if get_mask:
                targets = targets.to(device).unsqueeze(1)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    total_loss/= len(loader.dataset)
    return total_loss

def set_train(configuration):
    """
    Funcion para obtener el device y batch size del entrenamiento
    """
    IMG_HEIGHT = configuration["image"]["height"]
    IMG_WIDTH = configuration["image"]["width"]
    verbose = configuration["train"]["verbose"]

    device = select_device()
    if verbose:
        print(f"Utilizando {device}")
    configuration["train"]["device"] = device
    
    # Select Batch Size
    if device == "cuda":
        gpu_ram = configuration["train"]["gpu ram"]
        configuration["train"]["batch size"] = gpu_ram//(IMG_HEIGHT*IMG_WIDTH)
    if verbose:
        print(f"batch size: {configuration["train"]["batch size"]}")
    
    return configuration

def set_model(BASE_DIR, configuration, selected_model):
    """
    Fucnion para obtener el modelo con todos sus parametros seteados
    """
    MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
    IMG_HEIGHT = configuration["image"]["height"]
    IMG_WIDTH = configuration["image"]["width"]
    model_name = configuration["models"][selected_model]["model"]
    criterion_configuration = configuration["models"][selected_model]["criterion"]
    optimizer_configuration = configuration["models"][selected_model]["optimizer"]
    verbose = configuration["train"]["verbose"]
    device = configuration["train"]["device"]
    n_classes = configuration["train"]["classes"]
    use_saved_model = configuration["train"]["use saved"]
    dropout_rate = configuration["train"]["dropout"]
    lr = configuration["train"]["learn rate"]

    if verbose:
        print(f"Modelo: {selected_model}")
    
    if use_saved_model:
        use_saved_model = os.path.exists(os.path.join(MODEL_PATH, f"{selected_model}.pth"))

    model = getattr(models, model_name[0])(dropout_rate = dropout_rate, out_classes=n_classes, img_heigth=IMG_HEIGHT, img_width=IMG_WIDTH)
    if use_saved_model:
        if verbose:
            print("Usando salvado")
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth"), map_location=torch.device(device)))
    else:
        if len(model_name) != 1:
            try:  
                aux = getattr(models, model_name[1])(dropout_rate = dropout_rate, out_classes=n_classes, img_heigth=IMG_HEIGHT, img_width=IMG_WIDTH)
                aux.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name[1]}.pth"), map_location=torch.device(device)))
                model.encoder.load_state_dict(aux.encoder.state_dict())
            except: "Modelo de transfer learning no ecnontrado"

    if len(model_name) != 1:
        for param in model.encoder.parameters():
            param.requires_grad = False
    model.to(device)

    criterion = getattr(torch.nn, criterion_configuration)()
    optimizer = getattr(torch.optim, optimizer_configuration)(model.parameters(), lr=lr)

    return model, criterion, optimizer

def train_loop(BASE_DIR, configuration, selected_model, model, optimizer, criterion, train_loader, validation_loader):
    """
    Bucle de entrenamiento principal
    """
    MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
    isClasification = configuration["train"]["is clasification"]
    verbose = configuration["train"]["verbose"]
    device = configuration["train"]["device"]
    n_epochs = configuration["train"]["epochs"]
    patience = configuration["train"]["patience"]
    dropout_rate = configuration["train"]["dropout"]
    print_epoch = configuration["train"]["print epoch"]
    get_mask = configuration["models"][selected_model]["get mask"]
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
    selected_model_state_dict = model.state_dict()
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
    save_json(BASE_DIR, configuration, selected_model, train_losses, validation_losses)

    if verbose:
        print('La mejor Epoca fue {:03d}, Train loss: {:.4f}, Validation loss: {:.4f}'.format(best_epoch, best_train_loss, best_val_loss))
        end = time.time()
        elapsed = end-start
        print(f"Tiempo de ejecución: {int(elapsed // 3600)} horas, {int((elapsed // 60) % 60)} minutos, {int(elapsed % 60)} segundos")

    return model

def separate(configuration):
    """
    Funcion auxiliar para separar los entrenamientos sin y con mascara
    """
    auto = []
    segmentation = []

    model_options = configuration["models"]["all"]

    for selected_model in model_options:
        if configuration["models"][selected_model]["get mask"]:
            segmentation.append(selected_model)
        else:
            auto.append(selected_model)
    
    return auto, segmentation

def dice(mask1, mask2):
    mask1 = (mask1 > 0.5)
    mask2 = (mask2 > 0.5)
    
    intersection = np.logical_and(mask1, mask2).sum()
    sumatoria = (mask1.sum() + mask2.sum())
    if sumatoria > 0.1:
        return 2. * intersection / (mask1.sum() + mask2.sum())
    else:
        return 0.0

def jaccard(mask1, mask2):
    mask1 = (mask1 > 0.5)
    mask2 = (mask2 > 0.5)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union > 0.1:
        return intersection / union
    else:
        return 0.0

def test_segmentation(BASE_DIR, configuration, selected_model, model, loader):
    """
    Testear segmentacion
    """
    HITO_PATH = configuration["path"]["hito"]
    verbose = configuration["train"]["verbose"]
    device = configuration["train"]["device"]
    batch_size = configuration["train"]["batch size"]
    idx = configuration["test"]["idx"]
    get_mask = configuration["models"][selected_model]["get mask"]

    idx_loader = idx//batch_size
    idx = idx % batch_size
    try:
        model.eval()
        i = 0
        for data in loader:
            if i == idx_loader:
                inputs, targets = data
                with torch.no_grad():
                    outputs = model(inputs.to(device))
                input = inputs[idx]
                output = outputs[idx]
                target = targets[idx]
            i += 1
        input_image = input.permute(1, 2, 0).cpu().numpy()/255
        if get_mask:
            target_image = target.squeeze().cpu().numpy()
            output_image = output.squeeze().cpu().numpy()
        else:
            target_image = target.permute(1, 2, 0).cpu().numpy()/255
            output_image = output.permute(1, 2, 0).cpu().numpy()/255
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_image)
        axs[0].set_title("Input")

        axs[1].imshow(target_image, cmap='gray' if get_mask else None)
        axs[1].set_title("Truth")

        axs[2].imshow(output_image, cmap='gray' if get_mask else None)
        axs[2].set_title("Output")

        for ax in axs:
            ax.axis('off')

        fig.suptitle(f"{selected_model}")

        plt.tight_layout()
        if verbose:
            plt.show()
        fig.savefig(os.path.join(BASE_DIR, HITO_PATH, f"{selected_model}.png"))
    except: print("Imagen exede el batch")

    if get_mask:
        n_dice = 0
        n_jaccard = 0
        total = 0
        for data in loader:
            inputs, targets = data
            with torch.no_grad():
                outputs = model(inputs.to(device))
            outputs = outputs.squeeze().cpu().numpy()
            targets = targets.squeeze().cpu().numpy()
            for i in range(len(outputs)):
                total += 1
                n_dice += dice(outputs[i], targets[i])
                n_jaccard += jaccard(outputs[i], targets[i])
        
        print(f"dice: {n_dice}")
        print(f"jaccard: {n_jaccard}")
        hito = {"dice": n_dice/total, "jaccard": n_jaccard/total}
        JSON_FILE = os.path.join(BASE_DIR, HITO_PATH, f"{selected_model}.json")
        with open(JSON_FILE, 'w') as file:
            json.dump(hito, file, indent=4)


def test_clasification(BASE_DIR, configuration, selected_model, model, loader):
    """
    Testear segmentacion
    """
    HITO_PATH = configuration["path"]["hito"]
    verbose = configuration["train"]["verbose"]
    device = configuration["train"]["device"]
    n_classes = configuration["train"]["classes"]

    correct = 0
    output_result = np.zeros(n_classes)
    target_result = np.zeros(n_classes)
    total = 0

    model.eval()
    for data in loader:
        inputs, targets = data
        with torch.no_grad():
            outputs = model(inputs.to(device))
            outputs = outputs.cpu()
            for i in range(len(outputs)):
                total += 1
                if (torch.argmax(outputs[i]).item() == targets[i].item()):
                    correct += 1
                output_result[torch.argmax(outputs[i]).item()] += 1
                target_result[targets[i].item()] += 1
    
    correct = correct/total
    output_result = output_result/total
    target_result = target_result/total

    if verbose:
        print(f"Correct: {correct}")
        print(f"Output result:")
        for i in range(n_classes):
            print(f"class {i}: {output_result[i]}")
        print(f"Output target:")
        for i in range(n_classes):
            print(f"class {i}: {target_result[i]}")
    
    hito = {"correct": correct, "output result": output_result.tolist(), "target result": target_result.tolist()}
    JSON_FILE = os.path.join(BASE_DIR, HITO_PATH, f"{selected_model}.json")
    with open(JSON_FILE, 'w') as file:
        json.dump(hito, file, indent=4)


def graph(BASE_DIR):
    """
    Funcion auxiliar encargada de graficar la evolucion del ultimo entrenamiento realizado de cada modelo
    """
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

    model_options = configuration["models"]["all"]
    HISTORY_PATH = configuration["path"]["history"]
    HITO_PATH = configuration["path"]["hito"]

    for selected_model in model_options:

        JSON_FILE = os.path.join(BASE_DIR, HISTORY_PATH, f"{selected_model}.json")

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

            plt.savefig(os.path.join(BASE_DIR, HITO_PATH, f"{selected_model}_loss.png"))

        else:
            print(f"json file {JSON_FILE} not found")