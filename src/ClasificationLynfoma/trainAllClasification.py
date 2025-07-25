"""
Module to train all clasification models in this directory
"""

import os
import json
from src import utils
from src.ClasificationLynfoma import load

def main():
    """
    Run code
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad =  [True, True, False]

    with open(os.path.join(BASE_DIR, 'configurationClasification.json')) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)

    train_loader, validation_loader, _ = load.get_loaders(configuration, toLoad)
    for selected_model in configuration["models"]["all"]:
        model, criterion, optimizer = utils.set_model(BASE_DIR, configuration, selected_model)
        model = utils.train_loop(BASE_DIR, configuration, selected_model, model, optimizer, criterion, train_loader, validation_loader)


if __name__ == "__main__":
    main()