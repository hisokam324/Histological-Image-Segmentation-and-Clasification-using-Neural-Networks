"""
Auxiliary module to create loaders
"""

from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loaders(configuration, toLoad):
    '''
    This function load PathMNIST images directly from MedMNIST and create the corresponding loaders
    
    Args:
        configuration (Dict): Configuration information, such as batch size
        
        toLoad (List[Boolean]): Indicates wich loaders to create in order Train, Vaidation and Test. False loaders are return empty
    
    Returns: 
        train_loader (PyTorch DataLoader): Loader meant for training
        
        validation_loader (PyTorch DataLoader): Loader meant for validation
        
        test_loader (PyTorch DataLoader): Loader meant for testing
    '''
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