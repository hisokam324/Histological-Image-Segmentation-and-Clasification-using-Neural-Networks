import json
import os
from src import utils
from src.Clasification import load

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    toLoad = [False, False, True]

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)
    
    configuration = utils.set_train(configuration)

    _, _, test_loader = load.get_loaders(configuration, toLoad)
    for selected_model in configuration["models"]["all"]:
        model, _, _= utils.set_model(BASE_DIR, configuration, selected_model)
        utils.test_clasification(BASE_DIR, configuration, selected_model, model, test_loader)

if __name__ == "__main__":
    main()