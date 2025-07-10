from medmnist import PathMNIST
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size):
    data_flag = 'pathmnist'
    download = True

    info = INFO[data_flag]
    DataClass = PathMNIST

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_dataset = DataClass(split='train', transform=data_transforms, download=download)
    val_dataset = DataClass(split='val', transform=data_transforms, download=download)
    test_dataset = DataClass(split='test', transform=data_transforms, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader