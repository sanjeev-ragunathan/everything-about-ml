'''
DCGAN
- Deep Convolutional Generative Adversarial Networks
'''

# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

import torch
import torch.nn as nn
import os

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        # out_channels - no.of filters
        # output_size = (input_size - kernel_size) / stride + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2) # MNIST is 28x28 pixel images, (28-3) / 2 + 1 = 13
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2) # (13-3) / 2 + 1 = 6
        self.fc = nn.Linear(6 * 6 * 64, 1)

    def forward(self, image):
        conv1_out = self.conv1(image)
        relu1_out = torch.relu(conv1_out)
        conv2_out = self.conv2(relu1_out)
        relu2_out = torch.relu(conv2_out)
        flatten_out = relu2_out.view(relu2_out.size(0), -1)
        fc_out = self.fc(flatten_out)
        sigmoid_out = torch.sigmoid(fc_out)
        
        return sigmoid_out

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 7 * 7 * 128)
        # output_size = (input_size - 1) × stride + kernel_size - 2 × padding
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) # (7-1)x2 + 4 - 2x1 = 14
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1) # (14-1)x2 + 4 - 2x1 = 28
    
    def forward(self, noise):
        fc_out = self.fc(noise)
        reshape_out = fc_out.view(fc_out.size(0), 128, 7, 7)
        conv_transpose1_out = self.conv_transpose1(reshape_out)
        relu1_out = torch.relu(conv_transpose1_out)
        conv_transpose2_out = self.conv_transpose2(relu1_out)
        tanh_out = torch.tanh(conv_transpose2_out)

        return tanh_out



if __name__ == "__main__":



    # DATASET
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

    
    
    # CREATE MODELS
    discriminator = Discriminator()
    generator = Generator()

    gen_path = "generator.pth"
    disc_path = "discriminator.pth"

    if os.path.exists(gen_path) and os.path.exists(disc_path):
        print("Loading saved models...")
        generator.load_state_dict(torch.load(gen_path))
        discriminator.load_state_dict(torch.load(disc_path))
    else:
        print("Training models...")

        # TRAIN

        # Loss function
        loss_fn = nn.BCELoss()
        # optimizer
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)

        # training loop
        for epoch in range(100):
            for real_images, _ in dataloader:
                # 1. Train Discriminator
                optimizer_D.zero_grad()
                
                # learn real images
                outputs_D_real = discriminator(real_images)
                loss_D_real = loss_fn(outputs_D_real, torch.ones_like(outputs_D_real))

                # learn fake images
                noise = torch.randn(real_images.size(0), 64) # use actual batch size instead of hardcoded 64
                fake_images_G = generator(noise)
                outputs_D_fake = discriminator(fake_images_G.detach()) # detach - prevent gradients flowing to generator during discriminator training
                loss_D_fake = loss_fn(outputs_D_fake, torch.zeros_like(outputs_D_fake))

                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()

                # 2. Train Generator
                optimizer_G.zero_grad()
                outputs_D_fake = discriminator(fake_images_G)  # no detach — gradients flow to generator
                loss_G = loss_fn(outputs_D_fake, torch.ones_like(outputs_D_fake))
                loss_G.backward()
                optimizer_G.step()
            print(f"Epoch {epoch}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)
        print("Models saved.")

    
    # GENERATE AND SAVE IMAGES
    import matplotlib.pyplot as plt

    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, 64)
        fake_images = generator(noise)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        img = fake_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Digits')
    plt.savefig('generated_digits.png')
    plt.close()
    print("Saved generated_digits.png")