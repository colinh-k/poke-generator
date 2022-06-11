import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm

def load_datasets(data_dir, image_w, image_h):
    train_transformation = transforms.Compose([
        transforms.Resize((image_w, image_h)),
        # transforms.RandomCrop((image_w, image_h), padding=8, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    return ImageFolder(data_dir, transform=train_transformation)

def train(dataloader, model_g, input_size, model_d, label_real, label_fake, criterion, optimizer_g, optimizer_d, epoch_start, n_epochs, device):
    # returns list of loss values for each epoch, and a list of sampled images from the generator
    losses_g = list()
    losses_d = list()
    sample_imgs = list()
    model_g.to(device)
    model_d.to(device)
    # optimizer_g.to(device)
    # optimizer_d.to(device)
    criterion.to(device)

    # record generated images from G using the same input vector at different points in training
    sample_noise = torch.randn(64, input_size, 1, 1, device=device)

    for epoch in range(epoch_start, n_epochs + epoch_start + 1):
        # for i, (inputs, labels) in enumerate(dataloader, 0):
        for i, (imgs, img_labels) in enumerate(tqdm(dataloader, desc=f'training epoch {epoch} of {n_epochs + epoch_start}')):
            imgs, img_labels = imgs.to(device), img_labels.to(device)

            ### phase 1: update discriminator
            model_d.zero_grad()

            batch_size = imgs.size(0)
            ## train with real batch
            labels_d = torch.full(
                (batch_size,), label_real, dtype=torch.float, device=device
            )
            output_d = model_d(imgs).view(-1)
            loss_d_real = criterion(output_d, labels_d)
            loss_d_real.backward()

            ## train with fake batch
            noise = torch.randn(batch_size, input_size, 1, 1, device=device)
            # generate fake images with generator
            generated = model_g(noise)
            labels_d.fill_(label_fake)
            # classify fake batch with discriminator
            output_d = model_d(generated.detach()).view(-1)
            loss_d_fake = criterion(output_d, labels_d)
            loss_d_fake.backward()
            optimizer_d.step()

            loss_d = loss_d_real + loss_d_fake

            ### phase 2: update generator
            model_g.zero_grad()

            ## we just generated images above, so use these to perform the forward pass
            labels_g = torch.full(
                (batch_size,), label_real, dtype=torch.float, device=device
            )
            output_g = model_d(generated).view(-1)
            loss_g = criterion(output_g, labels_g)
            loss_g.backward()
            optimizer_g.step()

            # tqdm.write(f'd loss: {loss_d}\ng loss: {loss_g}')

            losses_g.append(loss_g.item())
            losses_d.append(loss_d.item())

        # record sample generated images periodically, and at the end
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                gen_imgs = model_g(sample_noise).detach().cpu()
                sample_imgs.append((epoch, gen_imgs))

    return losses_g, losses_d, sample_imgs

# initialize generator and discriminator weights from a normal
# distribution with mean 0 and standard deviation 0.02. the weights
# in the convolution and batch norm layers are reinitialized
# after construction
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)