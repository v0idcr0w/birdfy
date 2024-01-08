import pathlib, os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms 

# custom dataset to load images

class BirdsDataset(Dataset): 
    def __init__(self, root: str, transform = None): 
        self.paths = list(pathlib.Path(root).glob("*/*"))  # paths to all imgs in target_dir
        self.classes = os.listdir(root) # list of class names (species)
        self.class_to_idx = {class_name:index for (index, class_name) in enumerate(self.classes)} 
        self.idx_to_class = {index:class_name for (index, class_name) in enumerate(self.classes)} # inverse of class_to_idx
        self.transform = transform 
    def __len__(self): 
        return len(self.paths) 
    def __getitem__(self, index: int):
        image_path = self.paths[index] # type PosixPath, or WindowsPath etc 
        image_pil = Image.open(image_path)
        image_class_idx = self.class_to_idx[image_path.parent.name]  
        if self.transform is not None: 
            image_tensor = self.transform(image_pil)
        else: 
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image_pil)
        return image_tensor, image_class_idx 

def fetch_data(batch_size: int, image_size: int, is_reduced: bool = True, num_workers: int = os.cpu_count()): 
    """
    Args:
        batch_size (int): number of images fed to the model per training iteration. 
        image_size (int): height and width of the images. 
        is_reduced (bool, optional): decides whether to use the full dataset or a reduced dataset. Defaults to True.
        num_workers (int, optional): choose the number of CPU cores. Defaults to the result yielded by os.cpu_count().
    """    
    # img path
    image_path = "data/" + "reduced" if is_reduced else "full"
    train_path = image_path + "/train"
    test_path = image_path + "/test"

    # transformation pipeline 
    data_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(), 
    ]) 

    train_data = BirdsDataset(root=train_path, transform=data_transform)
    test_data = BirdsDataset(root=test_path, transform=data_transform) 

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True) 
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_data, test_data, train_loader, test_loader