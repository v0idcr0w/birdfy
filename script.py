"""
Full script for automating training, saving a model etc. 
""" 
def main(): 
    import torch 

    # use gpu if one is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define constants 
    BATCH_SIZE = 1
    IMAGE_SIZE = 128

    # load dataset and dataloaders 
    from setup import fetch_data  

    train_data, test_data, train_dataloader, test_dataloader = fetch_data(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # load model and training function 
    from models import TinyVGG 
    from engine import train

    model = TinyVGG(in_shape=3, # number of color channels (3 for RGB) 
                    hidden_units=10, 
                    out_shape=len(train_data.classes)).to(device)

    # define optimizer and loss function 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    model_results = train(model=model, train_dataloader=train_dataloader, test_dataloader=train_dataloader, epochs=4, loss_fn=loss_fn, optimizer=optimizer, device=device)

    # save the model 
    from utils import save_model, save_results
    save_model(model, f"{model.name()}.pth")
    
    # save the results (for later plotting)
    save_results(model.name(), model_results)

import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main() 