from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

"""
Auxiliary module to load images
"""

def get_loaders(configuration, toLoad):
    '''
    This function load PathMNIST images directly from MedMNIST and create the corresponding loaders
    
    Args:
        configuration (dict): Configuration information, such as batch size
        
        toLoad (boolean list): Indicates wich loaders to load, none load leaders will be return empty
    
    Returns:
        Train loader (Loader): Loader meant for training
        
        Validation loader (Loader): Loader meant for validation
        
        Test loader (Loader): Loader meant for testing
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