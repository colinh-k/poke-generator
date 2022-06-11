from torch import nn

from train_utils import weights_init

class Discriminator(nn.Module):
    def __init__(self, n_channels, feature_size):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # input is (n_channels) x 64 x 64
            nn.Conv2d(n_channels, feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size. (feature_size) x 32 x 32
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # input size. (feature_size * 2) x 16 x 16
            nn.Conv2d(feature_size * 2, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input size. (feature_size * 4) x 8 x 8
            # nn.Conv2d(feature_size * 4, 1, 4, 2, 1, bias=False),
            nn.Conv2d(feature_size * 4, feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # input size. (feature_size * 8) x 4 x 4
            nn.Conv2d(feature_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)