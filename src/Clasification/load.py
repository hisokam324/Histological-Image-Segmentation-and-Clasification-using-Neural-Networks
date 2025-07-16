from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loaders(configuration, toLoad):
    batch_size = configuration["train"]["batch size"]
    download = True

    DataClass = PathMNIST

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    loaders = []
    split = ["train", "val", "test"]
    for i in range(3):
        if toLoad[i]:
            dataset = DataClass(split=split[i], transform=data_transforms, download=download)
            loaders.append(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False))
        else:
            loaders.append([])
    
    return loaders[0], loaders[1], loaders[2]