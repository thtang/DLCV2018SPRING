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

# import model structure
import acgan_model

random.seed(4)
torch.manual_seed(4)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[3]
pretrained_acgan = sys.argv[4]


##### ACGAN 
print("ACGAN inferencing ...")
# plot training log
with open("training_log/ACGAN_D_real_class_list.pkl", "rb") as f:
    D_real_class_list = pickle.load(f)
with open("training_log/ACGAN_D_fake_class_list.pkl", "rb") as f:
    D_fake_class_list = pickle.load(f)
with open("training_log/ACGAN_D_real_acc_list.pkl", "rb") as f:
    D_real_acc_list = pickle.load(f)
with open("training_log/ACGAN_D_fake_acc_list.pkl", "rb") as f:
    D_fake_acc_list = pickle.load(f)

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.plot(D_fake_class_list, label="fake classification loss")
plt.plot(D_real_class_list, label="real classification loss")
plt.title("Training loss of attribute classification")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(D_real_acc_list, label="D real_acc")
plt.plot(D_fake_acc_list, label="D fake_acc")
plt.title("Discriminator Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.savefig(os.path.join(output_dir,"fig3_2.jpg"))

# plot smile attribute
up = np.ones(10)
down = np.zeros(10)
fixed_class = np.hstack((up,down))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1)).cuda()

ACG = acgan_model.Generator().cuda()
ACG.load_state_dict(torch.load(pretrained_acgan))
ACG.eval()
fixed_img_output = ACG(fixed_input)
torchvision.utils.save_image(fixed_img_output.cpu().data, os.path.join(output_dir,"fig3_3.jpg"),nrow=10)

print("Done !")