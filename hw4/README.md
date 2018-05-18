# Image Generation and Feature Disentanglement
This folder contains the implementation of
* Variational Autoencoder (VAE)
* Generative Adversarial Network (GAN)
* Auxiliary Classifier Generative Adversarial Networks (ACGAN)
* InfoGAN

on part of CelebA dataset.

## Usage:
### Training:
Refers to the **[train](https://github.com/thtang/DLCV2018SPRING/tree/master/hw4/train)** folder.
### Testing:
```
bash hw4.sh [path to ./hw4_data/] [output folder]
```
or 
```
# VAE inference
python3 VAE_inference.py [path to ./hw4_data/] [output folder] [GPU id] [model path]

# GAN inference
python3 GAN_inference.py [path to ./hw4_data/] [output folder] [GPU id] [model path]

# ACGAN inference
python3 ACGAN_inference.py [path to ./hw4_data/] [output folder] [GPU id] [model path]
```
#### Dependency
`Python3` `pytorch==0.3.1` `torchvision==0.2.0` `scikit-learn` `skimage` `matplotlib`

## Results
#### Vational Autoencoder: *reconstruction*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/VAE_reconstruction.gif)
#### Vational Autoencoder: *random generation*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/VAE_random.gif)
#### Generative Adversial Network: *random generation*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/GAN.gif)
#### Auxiliary Classifier GAN: *smiling generation [Smile/No Smile]*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/ACGAN.gif)
#### InfoGAN: *Manipulating latent codes on part of CelebA (hairstyle)*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/InfoGAN.gif)

#### Reference
[1] https://github.com/thtang/ADLxMLDS2017/tree/master/hw4 <br>
[2] [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) <br>
[3] [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
](https://arxiv.org/abs/1606.03657)
