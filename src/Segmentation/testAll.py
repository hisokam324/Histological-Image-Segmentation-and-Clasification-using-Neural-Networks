import json
import os
from src import utils
from src.Segmentation import load

def body(configuration, models):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad = [False, False, True]
    _, _, test_loader = load.get_loaders(BASE_DIR, configuration, models[0], toLoad)
    for selected_model in models:
        model, _, _= utils.set_model(BASE_DIR, configuration, selected_model)
        utils.test_segmentation(BASE_DIR, configuration, selected_model, model, test_loader)
     
     

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)

    auto, segmentation = utils.separate(configuration)

    body(configuration, auto)
    body(configuration, segmentation)
    

if __name__ == "__main__":
    main()