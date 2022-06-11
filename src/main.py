from discriminator import Discriminator
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from pathlib import Path
import random

from plot_utils import plot_batch, plot_results
from train_utils import load_datasets, train
from generator import Generator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = Path(__file__).parent.parent.resolve()
DATA_ROOT = PATH / 'data'
POKES_DIR = DATA_ROOT / 'PokemonData'
GENERATED_DIR = DATA_ROOT / 'generated'
FIGURES_DIR = DATA_ROOT / 'figures'

# model parameters:
N_CHANNELS = 3
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# parameters for Generator and Discriminator
G_INPUT = 100
G_FEATURE_SIZE = 32
D_FEATURE_SIZE = 32

# constants for discriminator training
LABEL_FAKE = 0
LABEL_REAL = 1

# training parameters
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 2e-4

# define this file name if you want to use a checkpoint
CHECKPOINT_FNAME = DATA_ROOT / 'results.torch' # None

def main():
    # seed all rng's so we get reproducible results
    seed_everything()

    train_set = load_datasets(
        POKES_DIR, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )

    model_g = Generator(N_CHANNELS, G_INPUT, G_FEATURE_SIZE)
    model_d = Discriminator(N_CHANNELS, D_FEATURE_SIZE)
    model_g.to(DEVICE)
    model_d.to(DEVICE)

    optimizer_g = optim.Adam(model_g.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model_d.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # declare here, may be initialized to a number saved from a checkpoint
    epoch_start = 0

    if CHECKPOINT_FNAME is not None:
        # load from checkpoint
        checkpoint = torch.load(CHECKPOINT_FNAME)

        model_g.load_state_dict(checkpoint['model_g_state'])
        model_d.load_state_dict(checkpoint['model_d_state'])
        optimizer_g.load_state_dict(checkpoint['optim_g_state'])
        optimizer_d.load_state_dict(checkpoint['optim_d_state'])

        # epoch_start = checkpoint['epoch_start']
        epoch_start = 101

    losses_g, losses_d, sample_imgs = train(
        train_loader, model_g, G_INPUT, model_d, LABEL_REAL, LABEL_FAKE, criterion, optimizer_g, optimizer_d, epoch_start, N_EPOCHS, DEVICE
    )

    plot_results(
        range(len(losses_g)), losses_g, 'G', losses_d, 'D', 'Iteration', 'Loss', 'Training Losses for Generator and Discriminator', FIGURES_DIR / 'losses.png'
    )

    # with torch.no_grad():
    #     gen_imgs = model_g(noise_input).detach()
    for (i, gen_imgs) in sample_imgs:
        plot_batch(gen_imgs, f'Generated images after epoch {i}', FIGURES_DIR / f'epoch{i}gen.png')

    results = {
        'losses_g': losses_g,
        'model_g_state': model_g.state_dict(),
        'optim_g_state': optimizer_g.state_dict(),

        'losses_d': losses_d,
        'model_d_state': model_d.state_dict(),
        'optim_d_state': optimizer_d.state_dict(),

        'optimizer_name': 'Adam',
        'n_epochs': N_EPOCHS,
        'epoch_start': epoch_start,
        'image_size': (IMAGE_WIDTH, IMAGE_HEIGHT)
    }
    torch.save(results, DATA_ROOT / 'results2.torch')

    plot_batch(next(iter(train_loader))[0], 'Training Images', fname=FIGURES_DIR / 'training_images.png')

def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print(f'using device = {DEVICE}')
    main()