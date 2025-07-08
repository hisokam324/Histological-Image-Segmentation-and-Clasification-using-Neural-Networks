import torch
import os
from tqdm import tqdm
import json
from medmnist import PathMNIST
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader

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

def create_loader(batch_size = 64):
    data_flag = 'pathmnist'
    download = True

    info = INFO[data_flag]
    DataClass = PathMNIST

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_dataset = DataClass(split='train', transform=data_transforms, download=download)
    val_dataset = DataClass(split='val', transform=data_transforms, download=download)
    test_dataset = DataClass(split='test', transform=data_transforms, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def save_json(selected_model, train_losses, validation_losses, lr, n_epochs, patience, dropout_rate, use_saved_model):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

    JSON_PATH = configuration["path"]["history"]
    JSON_FILE = os.path.join(BASE_DIR, JSON_PATH, f"{selected_model}.json")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE) as file:
            history = json.load(file)  
    else:
        history = {}

    history[len(history)] = {"number of epochs": f"{len(validation_losses)}/{n_epochs}",
                    "patience": patience,
                    "learn rate": lr,
                    "dropout": dropout_rate,
                    "use saved model": use_saved_model,
                    "train loss": train_losses,
                    "validation loss" : validation_losses}

    with open(JSON_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def test_clasification(model, loader, device = "cuda", verbose = True):
    ceros = 0
    ones = 0
    twos = 0
    correct = 0
    total = 0
    model.eval()
    for data in tqdm(loader):
        inputs, targets = data
        with torch.no_grad():
            outputs = model(inputs.to(device))
            outputs = outputs.cpu()
            for i in range(len(outputs)):
                total += 1
                if (torch.argmax(outputs[i]).item() == targets[i].item()):
                    correct += 1
                if (torch.argmax(outputs[i]).item() == 0):
                    ceros += 1
                if (torch.argmax(outputs[i]).item() == 1):
                    ones += 1
                if (torch.argmax(outputs[i]).item() == 2):
                    twos += 1
    return correct/total, ceros/total, ones/total, twos/total