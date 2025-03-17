import torch

def test_model(model, test_dataloader):
 
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for data in test_dataloader:

            features, labels = data

            outputs = model(features)
            
            predicted = torch.argmax(outputs, dim=1)

            total += torch.numel(labels)

            correct += (predicted == labels).sum().item()

    return correct, total


