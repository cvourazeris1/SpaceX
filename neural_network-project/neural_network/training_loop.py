from neural_network import validation_loop
import torch

def train_model(model, train_dataloader, validate_dataloader, criterion, optimizer, num_epochs):

    # puts our model in training mode
    model.train()

    # create lists for training and validation loss per epoch
    train_epoch_loss = []
    validate_epoch_loss = []


    # loop through the number of epochs you want to train
    for epoch in range(num_epochs):
        
        # creates list of batch losses
        batch_losses = []

        # iterate through dataloader
        for data in train_dataloader:
            
            # unpack data from dataloader
            features, labels = data

            # go forward and get a prediction
            y_pred = model(features) 

            # measure the loss/error
            loss = criterion(y_pred, labels) # predicted values vs the y_train

            # neural network theory in practice
            optimizer.zero_grad() # clears the gradients
            loss.backward()       # computes derivates of loss function (gradient)
            optimizer.step()      # take step in direction of steepest descent of loss function

            batch_losses.append(loss.item())    # append loss for data in batch
        
        # keep Track of our losses
        train_epoch_loss.append(sum(batch_losses)/len(batch_losses))                        # calculates the average loss of all batches in an epoch
        val_loss = validation_loop.validate_model(model, validate_dataloader, criterion)    # calculates the validation loss of the model
        validate_epoch_loss.append(val_loss.item())                                         # saves validation loss after each epoch

        # print metrics
        print(f'Epoch: {epoch} Training Loss: {train_epoch_loss[epoch]} Validation Loss: {validate_epoch_loss[epoch]}')

    return model, train_epoch_loss, validate_epoch_loss
