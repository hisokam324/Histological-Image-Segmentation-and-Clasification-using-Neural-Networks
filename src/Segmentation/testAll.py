import json
import os
from src import utils
from src.Segmentation.test import test     

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    verbose = configuration["train"]["verbose"]
    model_options = configuration["models"]["all"]
    
    for selected_model in model_options:
        print(f"modelos: {selected_model}")
        model, MODEL_PATH, dropout_rate, device, get_mask, lr, criterion, optimizer, batch_size = utils.set_model(selected_model, configuration, BASE_DIR, verbose)
        test(model, selected_model, BASE_DIR, configuration, device, get_mask, verbose)

if __name__ == "__main__":
    main()