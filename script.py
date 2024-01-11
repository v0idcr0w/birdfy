"""
Full script for automating training, saving a model etc. 
How to use example: 
$ python script.py --batch_size 32 --image_size 224 --lr 0.001 --model_type effnetb0 --epochs 5 --name effnetb0 
""" 
import argparse 
import torch  
# custom modules
from engine import train
from setup import fetch_data  
from utils import save_model, save_results

def main(): 
    # use gpu if one is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description="Full script for training automation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--lr", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--model_type", type=str, default="tinyvgg", help="Model type")
    parser.add_argument("--name", type=str, help="Model name")

    # define constants 
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size 
    LEARNING_RATE = args.lr 
    EPOCHS = args.epochs 
    MODEL_NAME = args.name 
    MODEL_TYPE = args.model_type

    train_data, test_data, train_dataloader, test_dataloader = fetch_data(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    
    OUT_SHAPE = len(train_data.classes)
    
    if MODEL_TYPE == 'tinyvgg':
        from models import TinyVGG
        model = TinyVGG(in_shape=3, # number of color channels (3 for RGB) 
                    hidden_units=10, 
                    out_shape=OUT_SHAPE,
                    image_size=IMAGE_SIZE,
                    name=MODEL_NAME).to(device)
    else:
        from models import effnetb0_loader
        model = effnetb0_loader(out_shape=OUT_SHAPE, device=device)
        model.name = MODEL_NAME
    

    # define optimizer and loss function 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    model_results = train(model=model, train_dataloader=train_dataloader, test_dataloader=train_dataloader, epochs=EPOCHS, loss_fn=loss_fn, optimizer=optimizer, device=device)

    # save the model     
    save_model(model, f"{model.name}.pth")

    # save the results (for later plotting)
    save_results(model.name, model_results)

import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main() 