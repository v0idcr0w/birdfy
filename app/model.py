import torch, torchvision
from torch import nn

from PIL import Image 
import pathlib 

class TinyVGG(nn.Module): 
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
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
            nn.Linear(in_features=hidden_units*32*32, out_features=out_shape),
        ) 
        
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))  
    
    def name(self):
        return "TinyVGG"
    
labels_dict = {0: 'AUSTRALIAN MAGPIE', 1: 'HOUSE SPARROW', 2: 'NORTHERN CARDINAL'}

# define transformations
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(128, 128)), 
        torchvision.transforms.ToTensor(), 
    ])

def load_model(model_path: str):
    model = TinyVGG(in_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  out_shape=len(labels_dict)) 
    model_save_path = pathlib.Path(model_path) 
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict) 
    return model 

def get_prediction(file):
    image = transform(Image.open(file)).unsqueeze(0)  
    model = load_model("TinyVGG.pth")
    
    # put model in eval mode
    model.eval()
    with torch.inference_mode():
        preds = model(image)
    probs = torch.softmax(preds, dim=1)
    label_idx = torch.argmax(probs, dim=1).item() 
    prediction = labels_dict[label_idx] 
    # transform probs to dict 
    probs = probs.tolist()[0]
    probs_dict = {name.title() : round(prob * 100,1) for name, prob in zip(labels_dict.values(), probs)}
    return probs_dict, prediction.title() 
