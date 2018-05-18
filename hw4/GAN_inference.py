import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import skimage.io
import skimage
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from sklearn.manifold import TSNE

import pandas as pd

import warnings; warnings.simplefilter('ignore')
import sys
import os
import random
from os import listdir
import pickle
import gan_model

random.seed(4)
torch.manual_seed(4)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[3]
pretrained_gan = sys.argv[4]

print("GAN inferencing ...")
## plot training curve
with open("training_log/GAN_D_real_acc_list.pkl", "rb") as f:
    D_real_acc_list = pickle.load(f)
with open("training_log/GAN_D_fake_acc_list.pkl", "rb") as f:
    D_fake_acc_list = pickle.load(f)
with open("training_log/GAN_D_loss_list.pkl", "rb") as f:
    D_loss_list = pickle.load(f)
with open("training_log/GAN_G_loss_list.pkl", "rb") as f:
    G_loss_list = pickle.load(f)
# plot loss
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.plot(D_real_acc_list, label = "D real accuracy")
plt.plot(D_fake_acc_list, label = "D fake accuracy")
plt.title("Discriminator Accuracy")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(D_loss_list, label="D loss")
plt.plot(G_loss_list, label="G loss")
plt.title("Training Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig(os.path.join(output_dir,"fig2_2.jpg"))

# plot random generated image
fixed_noise = Variable(torch.randn(32, 100, 1, 1)).cuda()
G = gan_model.Generator().cuda()
G.load_state_dict(torch.load(pretrained_gan))
G.eval()
fixed_img_output = G(fixed_noise)
torchvision.utils.save_image(fixed_img_output.cpu().data, os.path.join(output_dir,"fig2_3.jpg"),nrow=8)