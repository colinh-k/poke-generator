# Poké-Generator

## Project Description
The goal of this project was to train a neural network to generate synthetic images of Pokémon. The project uses a generative adversarial network (GAN) to learn the dataset's general distribution. After training, the model generates new images based on this learned distribution.

## Existing Software Tools Used
- `pytorch` was used to implement the neural networks as well as load the dataset.
- `matplotlib` was used to visualize the resultant data.
- `tqdm` was used to provide progress updates during training.

## Dataset
This project uses a collection of about 7,000 images of Pokémon.<sup>[1](#Sources)</sup> There are about 25-50 images for each of the 150 types of Pokémon (from the first generation of games) included in the dataset. Each image contains a single Pokémon such that the subject is situated in the center of each image; however, each image may include a background scene not part of the Pokémon itself. When loading the dataset, the model applies a random horizontal flip to each image to increase variety. Further, each image is resized to dimensions 64x64.

## Model Implementation
GAN's are generally composed of two neural networks; the generator and discriminator. The generator is responsible for producing synthetic images from a randomly initialized input vector. The discriminator is responsible for distinguishing between actual images of Pokémon from the dataset and synthetic images produced by the generator. In this way, the discriminator is learning to classify images into a binary label set while providing feedback for the generator of how closely its synthetic images match the dataset samples. During training, the discriminator gets better at recognizing synthetic images while the generator gets better at producing 'realistic' synthetic images.

The generator and discriminator architectures are defined in `src/generator.py` and `src/discriminator.py`. The GAN architecture closely follows the GAN implementation provided by the `pytorch` tutorial series.<sup>[2](#Sources)</sup> 

### Generator
The first group of layers in the generator architecture uses a convolution layer followed by batch normalization and a ReLU activation function. This group is repeated three more times, where the input and output channels are adjusted in accordance with the kernel size and stride such that the final convolution layer yields dimensions 3x64x64. Finally, a tanh function is applied to squeeze the final layer output between -1 and 1.

### Discriminator
The discriminator architecture uses the same number of convolution layers. A leaky ReLU activation is applied after batch normalization between each convolution layer. Finally, a sigmoid activation function is applied on the final layer.

## Training
The criterion used during training to derive loss was binary cross entropy loss. Both models used independent instances of the Adam optimizer. The model was allowed to train for 200 epochs, which took approximately 110 minutes to run on Google Colab's GPU's. To ensure the results could be reproducible, all random number generators, including the ones used by pytorch, were seeded prior to training.

## Final Hyperparameters
- N epochs: 200
- Batch size: 128
- Learning rate: 2e-4
- Generator input size: 100
- Generator feature size: 64
- Discriminator feature size: 64

## Results
### Loss
The following plot shows the training loss curves vs batch iteration for both the generator and discriminator as they trained in parallel over 200 epochs. These networks are playing a 'zero-sum game', and so the loss curves reflect this jostling effect over the training period.
![Training losses for G and D](/data/figures/final_losses.png)

### Synthetic Images
During the training process, the generator was given the same randomly initialized input vector at the end of each epoch to produce 64 images. The results provide a benchmark for the model's performance over the training period. A selection of the generated images are shown below, and the complete set of collected images can be found in `data/figures`. Notice as the generator trained for longer, we begin to see the synthetic images take on shapes more similar to the training set than to random noise shown in the first image. I believe these images demonstrate the model has successfully learned from the dataset.
![Sample synthetic images](/data/figures/epoch0gen.png)
![Sample synthetic images](/data/figures/epoch50gen.png)
![Sample synthetic images](/data/figures/epoch100gen.png)
![Sample synthetic images](/data/figures/epoch150gen.png)
![Sample synthetic images](/data/figures/epoch200gen.png)

### Training Images
For reference, the following shows a random sample of 64 images from the training dataset.
![Sample synthetic images](/data/figures/training_images.png)

## Conclusions
In my opinion, the final images do not look like Pokémon; however, I believe it is remarkable how the final image sets produce continuous blobs of color gradients, despite the generator initially producing a random distribution of color in the images. Several of the images even have solid, single-color backgrounds behind the centered color blobs, which may indicate the model was improving its ability to distinguish background elements from the Pokémon present in the training set. 

To continue improving this model, I would experiment with different optimizers and training periods. As shown above, the model appears to perform better the longer it trains. Training time quickly became a limitation during the course of this project, and so I had to choose a range that gave interesting results, but could be completed in a reasonable amount of time. Lastly, I would attempt to find a larger dataset. With more training instances, the model may be able to learn more features of Pokémon and produce more 'realistic' and original images. Further, the art style of each generation of Pokémon is remarkably different, and so collecting images from other generations may produce a larger variety of Pokémon styles.

## Video Link
[Presentation video](https://youtu.be/kXEjGMVFw1o)

## Sources
1. [Pokémon dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) from Kaggle
2. [Pytorch GAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
