import matplotlib.pyplot as plt

def graph_loss(num_epochs, train_losses, validate_losses):

    plt.plot(range(num_epochs), train_losses)
    plt.plot(range(num_epochs), validate_losses)
    plt.legend()
    plt.xlabel('Epoch')