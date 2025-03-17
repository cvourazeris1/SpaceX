from neural_network import validation_loop
import torch

def train_model(model, train_dataloader, validate_dataloader, criterion, optimizer, num_epochs):
    # Train our model!
    model.train()

    # Epochs? (one run thru all the training data in our network)
    train_epoch_loss = []
    validate_epoch_loss = []

    for epoch in range(num_epochs):
        
        batch_losses = []

        for data in train_dataloader:

            features, labels = data

            # Go forward and get a prediction
            y_pred = model(features) # Get predicted results

            # Measure the loss/error, gonna be high at first
            loss = criterion(y_pred, labels) # predicted values vs the y_train

            # Do some back propagation: take the error rate of forward propagation and feed it back
            # thru the network to fine tune the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        
        # Keep Track of our losses
        train_epoch_loss.append(sum(batch_losses)/len(batch_losses))

        val_loss = validation_loop.validate_model(model, validate_dataloader, criterion)

        validate_epoch_loss.append(val_loss.item())


        print(f'Epoch: {epoch} Training Loss: {train_epoch_loss[epoch]} Validation Loss: {validate_epoch_loss[epoch]}')

    return model, train_epoch_loss, validate_epoch_loss
