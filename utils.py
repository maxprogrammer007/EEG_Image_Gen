import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_image_grid(images, path, nrow=5):
    fig, axs = plt.subplots(nrow, nrow, figsize=(8, 8))
    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            axs[i, j].imshow(images[idx], cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_training_metrics(epoch, d_losses, g_losses, rmse_list, c_losses, acc_list):
    
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(rmse_list, label='RMSE')
    plt.plot(c_losses, label='Category Loss')
    plt.plot(acc_list, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(f'comparison/loss_curve_epoch_{epoch}.png')
    plt.close()

def save_image_grid(images, path, nrow=5):
    
    fig, axs = plt.subplots(nrow, nrow, figsize=(8, 8))
    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            axs[i, j].imshow(images[idx], cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(epoch, y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'comparison/confusion_matrix_epoch_{epoch}.png')
    plt.close()

def save_image_comparison(real, fake, epoch, idx):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(real, cmap='gray')
    axs[0].set_title("Real")
    axs[0].axis("off")

    axs[1].imshow(fake, cmap='gray')
    axs[1].set_title("Generated")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(f'comparison/comparison_{epoch}_{idx}.png')
    plt.close()

def plot_losses(train_losses, test_losses, epochs,save_path=None):
    """
    Plots training and test losses.

    Parameters:
    - train_losses: List of training losses.
    - test_losses: List of test losses.
    - epochs: Number of epochs for the x-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compute_accuracy(pred, target):
    _, predicted = torch.max(pred.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct / total

from tqdm import trange, tqdm

#only for CNN 
def train(model, train_data, test_fea, test_label, optimizer, criterion, device, EPOCHS=1000, verbose=True):
    train_losses = []
    test_losses = []

    for epoch in trange(EPOCHS, desc="Epochs"):
        model.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 1, 10, 14)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_data))

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                inputs = test_fea.view(-1, 1, 10, 14).to(device)
                outputs, _ = model(inputs)
                test_loss = criterion(outputs, test_label.to(device)).item()
                test_acc = compute_accuracy(outputs, test_label.to(device))

                test_losses.append(test_loss)

                if verbose:
                    print(f"[{epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return train_losses, test_losses

def extract_features(model, train_fea, device, data_original, output_file_path="models/CNN_Features/features.pkl"):
    model.eval()
    all_features = []

    with torch.no_grad():
        for inputs in train_fea:  # data_loader is a list of numpy arrays
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1, 10, 14).to(device)
            features = model(inputs)[1]  # Get features from the first fully connected layer
            all_features.append(features.cpu().numpy())

    # Stack all feature vectors vertically
    EEG_train_features_from_cnn = np.vstack(all_features)

    # Concatenate CNN features with image features from the original data (140: is image part)
    all_new_data = np.hstack((EEG_train_features_from_cnn, data_original[:, 140:]))

    # Save the combined data as a pickle file
    with open(output_file_path, 'wb') as f:
        pickle.dump(all_new_data, f)

    print('Dumped as shape:', output_file_path, data_original.shape, all_new_data.shape)
