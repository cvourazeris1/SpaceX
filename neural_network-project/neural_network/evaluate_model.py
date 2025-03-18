import torch

def test_model(model, test_dataloader):
 
    correct = 0
    total = 0

    predictions_list = []
    labels_list = []
    outputs_list = []

    model.eval()

    with torch.no_grad():

        for data in test_dataloader:

            features, labels = data

            outputs = model(features)
            
            predicted = torch.argmax(outputs, dim=1)

            total += torch.numel(labels)

            correct += (predicted == labels).sum().item()

            predictions_list.append(predicted)
            labels_list.append(labels)
            outputs_list.append(outputs)

    total_predictions = torch.stack(predictions_list)
    total_labels = torch.stack(labels_list)
    total_outputs = torch.stack(outputs_list)

    return correct, total, total_predictions, total_labels, total_outputs


