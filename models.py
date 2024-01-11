"""Script where model architectures reside."""
import torch, torchvision
from torch import nn 

class TinyVGG(nn.Module): 
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int, image_size=224, name=None):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=hidden_units*(image_size//4)*(image_size//4), out_features=out_shape),
        )
        if name is not None: 
            self.name = name  
        else:
            self.name = "TinyVGG"
        
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))  
    
    def get_name(self):
        return self.name
    
def effnetb0_loader(out_shape:int, device="cpu"):
    model = torchvision.models.efficientnet_b0(weights='DEFAULT').to(device)
    for param in model.features.parameters():
        # turn off gradient tracking  
        param.requires_grad = False 

    # change out_features for the classifier head
    model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=out_shape)
    ).to(device)
    return model 