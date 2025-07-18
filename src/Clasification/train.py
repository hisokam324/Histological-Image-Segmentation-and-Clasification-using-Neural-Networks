"""
Module to train one model in this directory
"""

import os
import json
from src import utils
from src.Clasification import load

def main():
    """
    Run code
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad =  [True, True, False]

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)

    selected_model = utils.select_model(configuration)
    train_loader, validation_loader, _ = load.get_loaders(configuration, toLoad)
    model, criterion, optimizer = utils.set_model(BASE_DIR, configuration, selected_model)
    model = utils.train_loop(BASE_DIR, configuration, selected_model, model, optimizer, criterion, train_loader, validation_loader)


if __name__ == "__main__":
    main()