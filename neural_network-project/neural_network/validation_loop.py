import torch

def validate_model(model, validate_dataloader, criterion):
 
    model.eval()

    with torch.no_grad():

        for data in validate_dataloader:

            features, labels = data

            outputs = model(features)
            
            val_loss = criterion(outputs, labels)

    return val_loss
