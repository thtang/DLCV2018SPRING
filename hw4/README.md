# Image Generation and Feature Disentanglement



## Usage:

```
python3 hw4_inference.py [GPU ID] [model path] [path to ./hw4_data/] [output folder]
```
#### Dependency
`pytorch==0.3.1` `torchvision==0.2.0` `scikit learn` `skimage`

## Baseline Performance
#### Vational Autoencoder *reconstruction*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/VAE_reconstruction.gif)
#### Vational Autoencoder *random generation*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/VAE_random.gif)
#### Generative Adversial Network *random generation*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/GAN.gif)
#### Auxiliary Classifier GAN *smiling generation*
![Alt Text](https://github.com/thtang/DLCV2018SPRING/blob/master/hw4/gif/GAN.gif)
