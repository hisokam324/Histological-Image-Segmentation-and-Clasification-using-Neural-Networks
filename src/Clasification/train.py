import os
import json
from src import utils
from src.Clasification import load

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    HISTORY_PATH = os.path.join(BASE_DIR, configuration["path"]["history"])

    n_epochs, patience, print_epoch = utils.set_train(configuration)

    selected_model = utils.select_model(model_options, configuration, verbose)
    model, criterion, optimizer = utils.set_model(BASE_DIR, configuration, selected_model)
    train_loader, validation_loader, _ = load.get_loaders(batch_size)
    model = utils.train_loop(selected_model, MODEL_PATH, model, optimizer, criterion, train_loader, validation_loader, n_epochs, patience, dropout_rate, lr, device, isClasification, get_mask, HISTORY_PATH, verbose, print_epoch)


if __name__ == "__main__":
    main()