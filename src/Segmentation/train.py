import os
import json
from src import utils
from src.Segmentation import load

"""
Programa de entrenamiento basico.
Entrena a un modelo en particular, puede ser seleccionado por consola o preprogramado en configuracion.json
"""

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad =  [True, True, False]

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    configuration = utils.set_train(configuration)

    selected_model = utils.select_model(configuration)
    model, criterion, optimizer = utils.set_model(BASE_DIR, configuration, selected_model)
    train_loader, validation_loader, _ = load.get_loaders(BASE_DIR, configuration, selected_model, toLoad)
    model = utils.train_loop(BASE_DIR, configuration, selected_model, model, optimizer, criterion, train_loader, validation_loader)


if __name__ == "__main__":
    main()