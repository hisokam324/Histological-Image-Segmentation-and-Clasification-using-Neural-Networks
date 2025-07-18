"""
Module to test one model in this directory
"""

import json
import os
from src import utils
from src.Segmentation import load  

def main():
    """
    Run code
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad = [False, False, True]

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)

    selected_model = utils.select_model(configuration)
    _, _, test_loader = load.get_loaders(configuration, selected_model, toLoad)
    model, _, _= utils.set_model(BASE_DIR, configuration, selected_model)
    utils.test_segmentation(BASE_DIR, configuration, selected_model, model, test_loader)

if __name__ == "__main__":
    main()