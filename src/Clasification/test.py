import torch
import json
import os
import matplotlib.pyplot as plt
from src import utils
from src.Segmentation import load

def test(model, selected_model, BASE_DIR, configuration, device, get_mask, verbose):
    DATA_PATH = os.path.join(BASE_DIR, configuration["path"]["data"])
    data_division = configuration["path"]["data division"]
    
    X, Y = load.load_images(os.path.join(DATA_PATH, data_division[2]), configuration, get_mask, verbose)

    model.eval()
    idx = 0
    input = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
    target = Y[idx]
    with torch.no_grad():
        output = model(input)


    input_image = input.squeeze().permute(1, 2, 0).cpu().numpy() / 255
    if get_mask:
        target_image = target.squeeze()
        output = (output > 0.5)
        output_image = output.squeeze().cpu().numpy()
    else:
        target_image = target.transpose(1, 2, 0)
        output_image = output.squeeze().permute(1, 2, 0).cpu().numpy() / 255


    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_image)
    axs[0].set_title("Input")

    axs[1].imshow(target_image, cmap='gray' if get_mask else None)
    axs[1].set_title("Truth")

    axs[2].imshow(output_image, cmap='gray' if get_mask else None)
    axs[2].set_title("Output")

    for ax in axs:
        ax.axis('off')

    fig.suptitle(f"{selected_model}")

    plt.tight_layout()
    if verbose:
        plt.show()
    fig.savefig(os.path.join(BASE_DIR, "hito", f"{selected_model}.png"))
     

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
            configuration = json.load(file)

    verbose = configuration["train"]["verbose"]
    model_options = configuration["models"]["all"]

    selected_model = utils.select_model(model_options, configuration, verbose)

    model, MODEL_PATH, dropout_rate, device, get_mask, lr, criterion, optimizer, batch_size = utils.set_model(selected_model, configuration, BASE_DIR, verbose)
    test(model, selected_model, BASE_DIR, configuration, device, get_mask, verbose)

if __name__ == "__main__":
    main()