import torch 
import torchvision
"""This module contains utility functions for fitting a PyTorch model"""
 
def train_step(model, dataloader, loss_fn, optimizer, device="cpu"): 
    model.train() 
    train_loss, train_acc = 0, 0 
    
    # loop through dataloader in batches 
    for batch, (X, y) in enumerate(dataloader): 
        X, y = X.to(device), y.to(device) 
        
        # forward pass 
        y_pred = model(X) 
        
        # compute loss 
        loss = loss_fn(y_pred, y) 
        train_loss += loss.item() 
        
        # zero gradients 
        optimizer.zero_grad() 
        
        # backward pass 
        loss.backward() 
        optimizer.step() 
        
        # computing acc
        y_pred_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1) 
        train_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels) 
    
    # scale metrics down 
    train_loss /= len(dataloader)
    train_acc /= len(dataloader) 
    
    return train_loss, train_acc 


def test_step(model, dataloader, loss_fn, device="cpu"): 
    model.eval() 
    
    test_loss, test_acc = 0, 0 
    
    with torch.inference_mode(): 
        for batch, (X, y) in enumerate(dataloader): 
            X, y = X.to(device), y.to(device)
            
            # forward pass 
            test_logits = model(X) 
            
            # compute loss
            loss = loss_fn(test_logits, y) 
            test_loss += loss.item() 
            
            # compute acc
            test_pred_labels = torch.argmax(torch.softmax(test_logits, dim=1), dim=1) 
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels) 
    # scale back metrics
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, epochs, loss_fn, optimizer, device="cpu"): 
    results = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []} 
    for epoch in range(epochs): 
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device) 
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        
        # progress
        print(f"***** EPOCH {epoch} *****")
        print(f"LOSS: TRAIN = {train_loss:.3f} | TEST = {test_loss:.3f}")
        print(f"ACCURACY: TRAIN = {train_acc:.3f} | TEST = {test_acc:.3f}")
        
        # add to results 
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
        
    return results 

def predict(model, image_tensor, transform, labels_dict, device="cpu"): 
    custom_image = transform(image_tensor).unsqueeze(dim=0) # add another dimension that corresponds to the batch size  
    
    model.eval() 
    with torch.inference_mode():
        preds = model(custom_image.to(device))
    probs = torch.softmax(preds, dim=1)
    label_idx = torch.argmax(probs, dim=1).item() 
    prediction = labels_dict[label_idx] 
    print(f"Predicted label: {prediction} with { torch.max(probs).item() * 100:.2f}% probability.")
    return probs, prediction 
    
    
