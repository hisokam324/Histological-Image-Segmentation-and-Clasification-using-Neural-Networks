import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import utils
import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'configuration.json')) as file:
        configuration = json.load(file)

verbose = True
model_options = configuration["models"]["all"]

if verbose:
    selecting_model = True
    while selecting_model:
        try:
            print("Selec model:")
            for i in range(len(model_options)):
                print(f"{i} {model_options[i]}")
            selected = int(input("modelo: "))
            selected = selected%len(model_options)
            selecting_model = False
        except:
            print("Modelo invalido, por favor ingrede un numero (integer)")
    selected_model = model_options[selected]
    print(f"modelo seleccionado: {selected} {selected_model}")
else :
    selected = configuration["train"]["select model"]
    selected_model = model_options[selected]

MODEL_PATH = os.path.join(BASE_DIR, configuration["path"]["models"])
DATA_PATH = os.path.join(BASE_DIR,configuration["path"]["data"])
data_division = configuration["path"]["data division"]
IMG_HEIGHT = configuration["image"]["heigth"]
IMG_WIDTH = configuration["image"]["width"]
get_mask = configuration["models"][selected_model]["get mask"]
model_name = configuration["models"][selected_model]["model"]
criterion_configuration = configuration["models"][selected_model]["criterion"]
optimizer_configuration = configuration["models"][selected_model]["optimizer"]
use_saved_model = configuration["train"]["use saved"]
dropout_rate = configuration["train"]["dropout"]
lr = configuration["train"]["learn rate"]
n_epochs = configuration["train"]["epochs"]
patience = configuration["train"]["patience"]
print_epoch = configuration["train"]["print epoch"]

device = utils.select_device(verbose)
if device == torch.device("cuda"):
    gpu_ram = configuration["train"]["gpu ram"]
    batch_size = int(gpu_ram/(IMG_HEIGHT*IMG_WIDTH))
else:
    batch_size = configuration["train"]["batch size"]

X, Y = utils.load_images(os.path.join(DATA_PATH, data_division[2]), get_mask)

model = getattr(models, model_name[0])(dropout_rate= dropout_rate)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{selected_model}.pth")))
model.to(device)

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
plt.show()
fig.savefig(os.path.join(BASE_DIR, "hito", f"{selected_model}.png"))