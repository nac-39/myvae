from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import torchvision
import torch
from model import VAE, Params
from argparse import ArgumentParser

def main():
    params = Params()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = VAE(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(params.epochs):
        loss_sum = 0.0
        cnt = 0
        for x, label in dataloader:
            x = x.to(params.device)
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        if epoch % 10 == 0:
            print(f"epoch: {epoch}/{params.epochs}, loss_avg: {loss_avg}")

    # Save the trained model
    torch.save(model.state_dict(), "vae_model.pth")
    print("Trained model saved to vae_model.pth")


def generate_samples():
    params = Params()
    model = VAE(params)
    model.load_state_dict(torch.load("vae_model.pth"))
    model.to(params.device)
    model.eval()
    with torch.no_grad():
        sample_size = 64
        z = torch.randn(sample_size, params.latent_dim).to(params.device)
        x = model.decoder(z)
        if x.device != 'cpu':
            x = x.cpu()
        generated_images = x.view(sample_size, 1, 28, 28)
    grid_img = torchvision.utils.make_grid(
        generated_images, nrow=8, padding=2, normalize=True
    )
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    utils.save_image(generated_images, 'generated_images.png')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate images from a trained model instead of training.",
    )
    args = parser.parse_args()

    if args.generate:
        generate_samples()
    else:
        main()
