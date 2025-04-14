import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.GAN import *
from models.CNNClassifier import *
from utils import *
from tqdm import tqdm

# Create output directories
os.makedirs('generated_images', exist_ok=True)
os.makedirs('comparison', exist_ok=True)
os.makedirs('activations', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

img_size = 40 * 56
noise_size = 60
n_class = 5
batch_size = 80
epochs = 1000
smooth = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pickle.load(open(r'models\CNN_Features\features.pkl', 'rb'))
label = data[:, -1]
data = data[:, :-1]
data = torch.tensor(data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.long)

# Dataset
dataset = TensorDataset(
    data[:, 40:].clone().detach().float(),
    data[:, :40].clone().detach().float(),
    label.clone().detach().long()
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(noise_size, n_class).to(device)
discriminator = Discriminator(n_class).to(device)

# Hook to capture activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

discriminator.conv1.register_forward_hook(get_activation('conv1_d'))
generator.deconv1.register_forward_hook(get_activation('deconv1_g'))

# Optimizers
lr = 0.0002
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
CrossEntropyLoss = nn.CrossEntropyLoss()

d_losses, g_losses, rmse_list, acc_list, c_losses = [], [], [], [], []

# Training loop
print(f"Generator is on: {next(generator.parameters()).device}")
print(f"Discriminator is on: {next(discriminator.parameters()).device}")
print(f"Training Started using device : {device}")

# === Training Loop ===
for epoch in tqdm(range(epochs)):
    for real_img, EEG, ground_truth in dataloader:
        batch_noise = torch.cat((torch.randn(batch_size, 20).to(device), EEG.to(device)), 1)

        # Train discriminator
        discriminator.zero_grad()
        d_logits_real, _, real_category_pred = discriminator(real_img.view(-1, 1, 40, 56).to(device), batch_noise.to(device))
        d_logits_fake, _, fake_category_pred = discriminator(generator(batch_noise)[1].detach(), batch_noise.to(device))

        d_loss_real = F.binary_cross_entropy_with_logits(d_logits_real, torch.ones_like(d_logits_real).to(device) * (1 - smooth))
        d_loss_fake = F.binary_cross_entropy_with_logits(d_logits_fake, torch.zeros_like(d_logits_fake).to(device))
        d_loss_rf = d_loss_real + d_loss_fake

        d_loss_category_real = F.binary_cross_entropy_with_logits(real_category_pred, F.one_hot(ground_truth, num_classes=5).float().to(device))
        d_loss = d_loss_rf + d_loss_category_real
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        for _ in range(2):
            g_optimizer.zero_grad()
            g_logits, g_outputs, pred = generator(batch_noise.to(device))
            d_logits_fake, _, fake_category_pred = discriminator(g_outputs, batch_noise.to(device))

            g_loss = F.binary_cross_entropy_with_logits(d_logits_fake, torch.ones_like(d_logits_fake).to(device))
            d_loss_category_fake = F.binary_cross_entropy_with_logits(fake_category_pred, F.one_hot(ground_truth, num_classes=5).float().to(device))
            g_regular = F.mse_loss(g_outputs.view(-1, 1), real_img.view(-1, 1).to(device))
            g_loss = g_loss + 0.8 * d_loss_category_fake + g_regular

            g_loss.backward()
            g_optimizer.step()

    # Logging and image saving
    if epoch % 100 == 0 and epoch != 0:
        train_loss_d = d_loss.item()
        train_loss_d_rf = d_loss_rf.item()
        train_loss_g = g_loss.item()
        train_loss_c = d_loss_category_fake.item()
        acc = compute_accuracy(fake_category_pred, ground_truth.to(device))
        ic_fake = compute_accuracy(fake_category_pred, ground_truth.to(device))
        ic_real = compute_accuracy(real_category_pred, ground_truth.to(device))
        d_losses.append(train_loss_d)
        g_losses.append(train_loss_g)
        rmse_list.append(g_regular.item())
        c_losses.append(train_loss_c)
        acc_list.append(acc)

        print(f"Epoch {epoch}/{epochs}, D Loss: {train_loss_d:.4f} (r/f: {train_loss_d_rf:.4f}), "
              f"G Loss: {train_loss_g:.4f}, RMSE: {g_regular:.4f}, C loss: {train_loss_c:.4f}, "
              f"acc: {acc:.4f}, IC real: {ic_real:.4f}, IC fake: {ic_fake:.4f}")

        plot_training_metrics(epoch, d_losses, g_losses, rmse_list, c_losses, acc_list)

        with torch.no_grad():
            gen_samples = generator(batch_noise.to(device))[1].view(-1, 40, 56).cpu().numpy()
            real_img = real_img.view(-1, 40, 56).cpu().numpy()

            for j in range(10):
                im = gen_samples[j]
                im = ((im + 1) * 127.5).astype(np.uint8)
                Image.fromarray(im, mode='L').save(f'generated_images/generated_{epoch}_{j}.jpg')

                rm = real_img[j]
                rm = ((rm + 1) * 127.5).astype(np.uint8)
                Image.fromarray(rm, mode='L').save(f'generated_images/real_{epoch}_{j}.jpg')

                save_image_comparison(rm, im, epoch, j)

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict()
        }, f'checkpoints/gan_checkpoint_epoch_{epoch}.pt')

        # Activation map visualization
        for i in range(min(3, g_outputs.size(0))):
            fmap = activations['deconv1_g'][i, 0].numpy()
            plt.imshow(fmap, cmap='viridis')
            plt.title(f'Generator Activation Map - Sample {i}')
            plt.colorbar()
            plt.savefig(f'activations/gen_activation_{epoch}_{i}.png')
            plt.close()

        # Confusion matrix
        plot_confusion_matrix(epoch, ground_truth.cpu(), fake_category_pred.argmax(dim=1).cpu())

        print('Images and activation maps saved')

