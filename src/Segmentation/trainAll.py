"""
Module to train all models in this directory
"""

import os
import json
from src import utils
from src.Segmentation import load

def body(configuration, models):
    """
    Auxiliary function to train some models in directory

    Args:
        configuration (Dict): Configuration information

        models (List[String]): List of models, keys from configuration dict
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad = [True, True, False]
    train_loader, validation_loader, _ = load.get_loaders(configuration, models[0], toLoad)
    for selected_model in models:
        model, criterion, optimizer = utils.set_model(BASE_DIR, configuration, selected_model)
        model = utils.train_loop(BASE_DIR, configuration, selected_model, model, optimizer, criterion, train_loader, validation_loader)
     

def main():
    """
    Run code:
        First: Auto Encoders

        Second: Segmentation Encoders
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    configuration = utils.set_train(configuration)

    auto, segmentation = utils.separate(configuration)
    body(configuration, auto)
    body(configuration, segmentation)

if __name__ == "__main__":
    main()