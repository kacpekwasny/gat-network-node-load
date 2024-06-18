import pandas as pd
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()


def plot_additional_metrics(train_losses, val_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, test_losses, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_results(train_losses, val_losses, test_losses):
    results = {
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Test Loss': test_losses
    }
    results_df = pd.DataFrame(results)
    print("Summary Statistics:\n")
    print(results_df.describe())

    plot_losses(train_losses, val_losses, test_losses)
    plot_additional_metrics(train_losses, val_losses, test_losses)
