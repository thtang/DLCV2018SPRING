import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import scipy.misc
import numpy as np
import skimage.transform
import skimage.io
from skimage.transform import resize

import warnings; warnings.simplefilter('ignore')
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
pretrained_weight_path = sys.argv[2]
input_dir = sys.argv[3]
output_dir = sys.argv[4]

class fcn32s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn32s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        # output_padding=0, groups=1, bias=True, dilation=1)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, num_classes, 64 , 32 , 0, bias=False),
        )
    def  forward (self, x) :        
        x = self.vgg.features(x)
#         print(x.size())
        x = self.vgg.classifier(x)
        return x

model = fcn32s(7).cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(pretrained_weight_path))
print("model loaded")
# $1: testing images directory (images are named 'xxxx_sat.jpg')
# $2: output images directory



# construct id list

image_path_list = sorted([file for file in os.listdir(input_dir) if file.endswith('.jpg')])
image_id_list = sorted(list(set([item.split("_")[0] for item in os.listdir(input_dir)])))

X = []

for i, file in enumerate(image_path_list):
    X.append(skimage.io.imread(os.path.join(input_dir, file)))
    

X = ((np.array(X)[:,::2,::2,:])/255).transpose(0,3,1,2)
print("X shape", X.shape)

X = torch.from_numpy(X).type(torch.FloatTensor)

# inference
model.eval()
pred = torch.FloatTensor()
pred = pred.cuda()

for i in range(len(X)):
    input_X = Variable(X[i].view(1,3,256,256).cuda())
    output = model(input_X)
    pred = torch.cat((pred,output.data),0)
pred = pred.cpu().numpy()
pred = np.argmax(pred,1)

print("resize...")
pred_512 = np.array([resize(p,output_shape=(512,512), order=0,preserve_range=True,clip=True) for p in pred])

print("decoding")
n_masks = len(X)
masks_RGB = np.empty((n_masks, 512, 512, 3))
for i, p in enumerate(pred_512):
    masks_RGB[i, p == 0] = [0,255,255]
    masks_RGB[i, p == 1] = [255,255,0]
    masks_RGB[i, p == 2] = [255,0,255]
    masks_RGB[i, p == 3] = [0,255,0]
    masks_RGB[i, p == 4] = [0,0,255]
    masks_RGB[i, p == 5] = [255,255,255]
    masks_RGB[i, p == 6] = [0,0,0]
masks_RGB = masks_RGB.astype(np.uint8)


print("save image...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, mask_RGB in enumerate(masks_RGB):
	skimage.io.imsave(os.path.join(output_dir,image_id_list[i]+"_mask.png"), mask_RGB)