from torch import nn

from train_utils import weights_init

class Generator(nn.Module):
    def __init__(self, n_channels, input_size, feature_size):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # in size: input_size x 1
            nn.ConvTranspose2d(input_size, feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(True),

            # in size: (feature_size * 8) x 4 x 4
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),

            # in size. (feature_size * 4) x 4 x 4
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),

            # in size. (feature_size * 2) x 8 x 8
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            # in size: 64 x 16 x 16
            nn.ConvTranspose2d(feature_size, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output size: n_channels x 32 x 32
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)