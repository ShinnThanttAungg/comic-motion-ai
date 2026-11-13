
import torch
import torch.nn as nn


class ComicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First convolution: 1 -> 8 channels
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second convolution: 8 -> 16 channels
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # def forward(self, x):
    # print("Input:", x.shape)

    # for layer in self.features:
    #     x = layer(x)
    #     print(
    #         layer.__class__.__name__,
    #         "->", x.shape
    #     )

    # x = self.classifier(x)
    # print("Classifier output:", x.shape)

    # return x


if __name__ == "__main__":
    model = ComicCNN()

    sample = torch.randn(4, 1, 32, 32)
    output = model(sample)

    print(model)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
