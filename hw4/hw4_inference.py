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
import vae_model
import gan_model
import acgan_model

random.seed(4)
torch.manual_seed(4)

os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
pretrained_vae = sys.argv[2]
pretrained_gan = sys.argv[3]
pretrained_acgan = sys.argv[4]
input_dir = sys.argv[5]
output_dir = sys.argv[6]


# VAE inference
test_img = []
test_folder = os.path.join(input_dir,"test")
test_images_path = sorted(listdir(test_folder))
for i in range(len(test_images_path)):
    img = skimage.io.imread(os.path.join(test_folder, test_images_path[i]))
    test_img.append(img)

test_img = np.array(test_img)/255*2-1
test_X = test_img.transpose(0,3,1,2)
test_X = torch.from_numpy(test_X).type(torch.FloatTensor)


model = vae_model.VAE(100).cuda()

model.load_state_dict(torch.load(pretrained_vae))
print("VAE model loaded")

# draw test image reconstruction 
model.eval()
pred = torch.FloatTensor()
pred = pred.cuda()
for i in range(len(test_X)):
    input_X_test = Variable(test_X[i].view(1,3,64,64).cuda())
    recon, mu, logvar = model(input_X_test)
    pred = torch.cat((pred,recon.data),0)
pred = pred.cpu()
draw_test = torch.cat((test_X[:10],pred[:10]),0) # visual first 10 reconstruct picture
torchvision.utils.save_image(draw_test/2+0.5, os.path.join(output_dir,"fig1_3.jpg"),nrow=10)
print("VAE recon saved")

rand_variable = Variable(torch.randn(32,100),volatile=True).cuda()
rand_output = model.decode(rand_variable)
torchvision.utils.save_image(rand_output.cpu().data/2+0.5, os.path.join(output_dir,"fig1_4.jpg"),nrow=8)
print("VAE random generate saved")

# store training log
with open("training_log/VAE_trainKLD.pkl", "rb") as f:
    train_KLD = pickle.load(f)
with open("training_log/VAE_trainMSE.pkl", "rb") as f:
    train_MSE = pickle.load(f)
    
# plot loss
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.plot(train_KLD)
plt.title("training KL divergence")
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(train_MSE)
plt.title("training MSE")
plt.xlabel("epoch")
plt.savefig(os.path.join(output_dir,"fig1_2.jpg"))

# visialize the latent space
input_X_test = Variable(test_X).cuda()
mu, logvar = model.encode(input_X_test)
latent_space = mu.cpu().data.numpy()
latent_embedded = TSNE(n_components=2, perplexity=30, random_state=4).fit_transform(latent_space)
print("tsne done")

# scatter plot 
test_attr = pd.read_csv(os.path.join(input_dir,"test.csv"))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
attr_arr = np.array(test_attr["Male"])

for i in [0,1]:
    if i==1:
        color="cornflowerblue"
        gender = "Male"
    else:
        color="hotpink"
        gender = "Female"
    xy = latent_embedded[attr_arr==i]
    label = attr_arr[attr_arr==i]
    ax1.scatter(xy[:,0], xy[:,1], c=color, label=gender)

ax1.legend()
ax1.set_title("Gender")
attr_arr = np.array(test_attr["Wearing_Lipstick"])

for i in [0,1]:
    if i==1:
        color="hotpink"
        gender = "Lipstick"
    else:
        color="steelblue"
        gender = "No Lipstick"
    xy = latent_embedded[attr_arr==i]
    label = attr_arr[attr_arr==i]
    ax2.scatter(xy[:,0], xy[:,1], c=color, label=gender)
ax2.legend()
ax2.set_title("Wearing Lipstick")
plt.savefig(os.path.join(output_dir,"fig1_5.jpg"))


##### GAN 
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