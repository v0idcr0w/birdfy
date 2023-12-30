import pathlib, os 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

def fetch_data(batch_size: int, image_size: int, is_reduced: bool = True, num_workers: int = os.cpu_count()): 
    """
    Args:
        batch_size (int): number of images fed to the model per training iteration. 
        image_size (int): height and width of the images. 
        is_reduced (bool, optional): decides whether to use the full dataset or a reduced dataset. Defaults to True.
        num_workers (int, optional): choose the number of CPU cores. Defaults to the result yielded by os.cpu_count().
    """    
    # img path
    image_path = pathlib.Path("data/" + "reduced" if is_reduced else "full")
    train_path = image_path / "train"
    test_path = image_path / "test"

    # transformation pipeline 
    data_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(), 
    ]) 

    train_data = datasets.ImageFolder(root=train_path, transform=data_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=test_path, transform=data_transform) 

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True) 
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_data, test_data, train_loader, test_loader


def idx_to_class(data): 
    """Returns the inverse of class_to_idx dictionary"""
    return {value:key for key, value in data.class_to_idx.items()} 