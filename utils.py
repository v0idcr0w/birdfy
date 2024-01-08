"""This module contains functionality for loading/saving models"""
import pathlib, os 
import torch 
import pandas as pd


def save_model(model, model_name: str):
    model_path = pathlib.Path("models")
    model_save_path = model_path / model_name
    torch.save(model.state_dict(), model_save_path)
    print("[INFO] Save successful")

def save_results(model_name, results):
    if not os.path.exists("model_results.csv"):
        with open("model_results.csv", 'w') as f:
            print("[INFO] Sucessfully created file")
            # add headers
            f.write("model_name|epoch_number|train_loss|test_loss|train_acc|test_acc\n")
    # append-mode
    with open("model_results.csv", "a") as f:
        for epoch in range(len(results['train_loss'])): 
            f.write(f"{model_name}|{epoch}|{results['train_loss'][epoch]}|{results['test_loss'][epoch]}|{results['train_acc'][epoch]}|{results['test_acc'][epoch]}\n")
        print("[INFO] Successfully written changes")
    
    # file structure
    """ 
    model_name|epoch_number|train_loss|test_loss|train_acc|test_acc
    """
            
         

def load_model(model, model_name: str): 
    model_path = pathlib.Path("models")
    model_save_path = model_path / model_name 
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict) 

def load_results(model_name: str | None = None):
    all_results = pd.read_csv("model_results.csv", delimiter="|", header=0)
    if model_name is None: 
        return all_results
    else: 
        return all_results[ all_results['model_name'] == model_name ]


if __name__ == "__main__":
    results = {"train_loss": [1.096, 0.688, 0.469, 0.301], 
               "test_loss": [0.977, 0.618, 0.532, 0.403],
               "train_acc": [0.515, 0.687, 0.801, 0.897],
               "test_acc": [0.491, 0.663, 0.793, 0.825]
               }
    model_name = "tinyvgg"
    # save_results(model_name, results) 
    print(load_results(model_name))
    

