# CGAN
An implementation of a Conditional Feed-Fordward Generative Adversarial Network. A Feed-Forward GAN is conditioned via an embedded vector of the label which is then concatenated with the input. For the Generator, the embedded vector is concatenated with the latent vector while in the Discriminator the embedded vector is concatenated with the flattened image and fed through the feed forward networks. The same GAN loss applies as a Vanilla GAN

Discriminator loss: 
