from torchvision import transforms, datasets
import torch
from model import VAE, Params


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    params = Params()
    model = VAE(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(params.epochs):
        loss_sum = 0.0
        cnt = 0
        for x, label in dataloader:
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f"epoch: {epoch}/{params.epochs}, loss_avg: {loss_avg}")


if __name__ == "__main__":
    main()
