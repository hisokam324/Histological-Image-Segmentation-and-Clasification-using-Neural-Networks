import os
import json
from src import utils
from src.Segmentation import load

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    isClasification = False

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    verbose = configuration["train"]["verbose"]
    model_options = configuration["models"]["all"]
    HISTORY_PATH = os.path.join(BASE_DIR, configuration["path"]["history"])

    n_epochs, patience, print_epoch = utils.set_train(configuration)

    for selected_model in model_options:
        if verbose:
            print(f"Modelo: {selected_model}")
        model, MODEL_PATH, dropout_rate, device, get_mask, lr, criterion, optimizer, batch_size = utils.set_model(selected_model, configuration, BASE_DIR, verbose)
        train_loader, validation_loader, _ = load.get_loaders(BASE_DIR, configuration, get_mask, batch_size, verbose)
        model = utils.train_loop(selected_model, MODEL_PATH, model, optimizer, criterion, train_loader, validation_loader, n_epochs, patience, dropout_rate, lr, device, isClasification, get_mask, HISTORY_PATH, verbose, print_epoch)

if __name__ == "__main__":
    main()