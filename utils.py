"""This module contains functionality for loading/saving models"""
import pathlib 
import torch 

def save_model(model, model_name: str):
    model_path = pathlib.Path("models")
    model_save_path = model_path / model_name
    torch.save(model.state_dict(), model_save_path)
    print("[INFO] Save successful")

def load_model(model, model_name: str): 
    model_path = pathlib.Path("models")
    model_save_path = model_path / model_name 
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict) 