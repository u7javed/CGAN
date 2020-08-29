# CGAN
An implementation of a Conditional Feed-Fordward Generative Adversarial Network. A Feed-Forward GAN is conditioned via an embedded vector of the label which is then concatenated with the input. For the Generator, the embedded vector is concatenated with the latent vector while in the Discriminator the embedded vector is concatenated with the flattened image and fed through the feed forward networks. The same GAN loss applies as a Vanilla GAN

Discriminator loss: max([Log(D(x|c) + Log(1-D(G(z|c)))]) Where D, G, z, c are Discriminator, Generator, latent vector, and label respectively.

Generator loss: min([Log(1 - D(G(z|c)))])

Python files:
  - models.py
    - Models.py contains the architecture for both the generator and discriminator. The vanilla GAN is consists of feed-forward generator and discriminator utilizing a leaky relu as its activation function. Tanh is used as the output activation for the generator while a sigmoid classification activation function is used for the discriminator.
    
  - train.py (EXECTUABLE SCRIPT)
    - train.py is an executable python script taking in various hyperparameters for training as arguments. This script loads the models and trains the models through a certain number of epochs.
    - The MNIST dataset is also automatically downloaded to a chosen directory.
    - Saves sample images from each epoch to specified directory
    - Saves models at each epoch to specified directory
