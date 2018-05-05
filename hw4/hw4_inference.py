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
from os import listdir
torch.manual_seed(38)

os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
pretrained_weight_path = sys.argv[2]
input_dir = sys.argv[3]
output_dir = sys.argv[4]

test_img = []
test_folder = os.path.join(input_dir,"test")
test_images_path = sorted(listdir(test_folder))
for i in range(len(test_images_path)):
    img = skimage.io.imread(os.path.join(test_folder, test_images_path[i]))
    test_img.append(img)

test_img = np.array(test_img)/255
test_X = test_img.transpose(0,3,1,2)
test_X = torch.from_numpy(test_X).type(torch.FloatTensor)
# define model
class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.latent_size = latent_size
        self.conv_stage = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
        )
        self.fcMean = nn.Linear(4096, self.latent_size)
        self.fcStd = nn.Linear(4096, self.latent_size)
        
        #decoding stage
        self.fcDecode = nn.Linear(self.latent_size,4096)
        
        self.trans_conv_stage = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128, 1.e-3),
            nn.LeakyReLU(0.01),
 
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64, 1.e-3),
            nn.LeakyReLU(0.01),
 
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32, 1.e-3),
            nn.LeakyReLU(0.01),
            
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1)
        )
        # final output activation function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def encode(self, x):
        conv_output = self.conv_stage(x).view(-1, 4096)
        return self.fcMean(conv_output), self.fcStd(conv_output)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()

        return eps.mul(std).add_(mu)


    def decode(self, z):
        fc_output = self.fcDecode(z).view(-1, 256, 4, 4)
#         print("decode fc output", fc_output.size())
        trans_conv_output = self.trans_conv_stage(fc_output)
#         print("trans_conv_output", trans_conv_output.size())
        return self.tanh(trans_conv_output)/2.0+0.5

    def forward(self, x):
        mu, logvar = self.encode(x)
#         print("mu shape",mu.size()," logvar",logvar.size())
        z = self.reparameterize(mu, logvar)
#         print("z shape",z.shape)
        return self.decode(z), mu, logvar

model = VAE(512).cuda()

model.load_state_dict(torch.load(pretrained_weight_path))
print("model loaded")
rand_variable = Variable(torch.randn(32,512),volatile=True).cuda()

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
torchvision.utils.save_image(draw_test, os.path.join(output_dir,"fig1_3.jpg"),nrow=10)

rand_output = model.decode(rand_variable)
torchvision.utils.save_image(rand_output.cpu().data, os.path.join(output_dir,"fig1_4.jpg"),nrow=8)
print("recon saved")




# visialize the latent space
input_X_test = Variable(test_X).cuda()
mu, logvar = model.encode(input_X_test)
latent_space = mu.cpu().data.numpy()
latent_embedded = TSNE(n_components=2, perplexity=30, random_state=38).fit_transform(latent_space)
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