import matplotlib.pyplot as plt

def graph_loss(num_epochs, train_losses, validate_losses):

    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), validate_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')