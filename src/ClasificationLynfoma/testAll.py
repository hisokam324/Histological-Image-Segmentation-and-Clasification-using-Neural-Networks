import json
import os
from src import utils
from src.ClasificationLynfoma import load

def main(name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad = [False, False, True]

    with open(os.path.join(BASE_DIR, f"{name}.json")) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)
    isClasification = configuration["train"]["is clasification"]

    _, _, test_loader = load.get_loaders(configuration, toLoad)
    for selected_model in configuration["models"]["all"]:
        model, _, _= utils.set_model(BASE_DIR, configuration, selected_model)
        if isClasification:
            utils.test_clasification(BASE_DIR, configuration, selected_model, model, test_loader)
        else:
            utils.test_segmentation(BASE_DIR, configuration, selected_model, model, test_loader)

if __name__ == "__main__":
    main("configurationSegmentation")
    main("configurationClasification")