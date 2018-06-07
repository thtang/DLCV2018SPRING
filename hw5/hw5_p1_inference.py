# hw5 prblem 1
from reader import readShortVideo
from reader import getVideoList
from os import listdir
import os
import sys
import pandas as pd
import numpy as np
import pickle

import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"]="0"

trimmed_valid_video_folder_path = sys.argv[1]
valid_gt_csv = sys.argv[2]
output_folder = sys.argv[3]
pre_trained_model_path = "models/CNN_FC_model.pkt"
### input ###
# normalize for pytorch model input
all_video_path = []
all_video_frame = []
video_path = trimmed_valid_video_folder_path
category_list = sorted(listdir(video_path))
for category in category_list:
    print(category)
    video_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
    a = ["-".join(file_name.split("-")[:5]) for file_name in video_list_per_folder]
    all_video_path += a
    for video in video_list_per_folder:
        frames = readShortVideo(video_path, category, video, downsample_factor=12, rescale_factor=1)
        all_video_frame.append(torch.stack(frames))

valid_X = []
for i in range(len(all_video_frame)):
    valid_X.append(all_video_frame[i])
    
### preparing labels
valid_df = pd.read_csv(valid_gt_csv)
valid_origin_order = valid_df["Video_name"].tolist()
valid_df = valid_df.sort_values(["Video_name"]).reset_index(drop=True)
valid_y = valid_df["Action_labels"].tolist()

# load pre-trained model and extract feature
cnn_feature_extractor = torchvision.models.densenet121(pretrained=True).features.cuda()
feature_size = 1024*7*7
cnn_feature_extractor.eval()
valid_features = []
counter = 0
with torch.no_grad():
    for i in range(len(valid_X)):
        input_X = valid_X[i]
        feature = cnn_feature_extractor(input_X.cuda()).cpu().view(-1, feature_size)
        valid_features.append(torch.mean(feature,0))
        counter +=1
        if counter % 100 == 0:
            print(counter)
valid_X = torch.stack(valid_features)
valid_y = torch.LongTensor(valid_y)
print("valid_X size", valid_X.size())

# basic classifier
class Net(torch.nn.Module):
    def __init__(self, feature_size):
        super(Net, self).__init__()
        
        self.linear1 = nn.Linear(feature_size,4096)
        self.linear2 = nn.Linear(4096,1024)
        self.linear3 = nn.Linear(1024, 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(4096)
        self.bn_2 = nn.BatchNorm1d(1024)
    def forward(self, x):
        x = self.relu(self.bn_1(self.linear1(x))) # same as relu output
        x = self.relu(self.bn_2(self.linear2(x)))
        y_pred = self.softmax(self.linear3(x))
        return y_pred
model = Net(feature_size).cuda()
model.load_state_dict(torch.load(pre_trained_model_path))
print("NN model loaded")
model.eval()
with torch.no_grad():
    output = model(valid_X.cuda())
    output_label = torch.argmax(output,1).cpu().data
    accuracy = np.mean((output_label == valid_y).numpy())
    print("validation accuracy: ",accuracy)
    
output_label = output_label.numpy()
# Since all_video_path is sort by Lexicon, we have to make the output with same order as in csv file 
reorder_index = []
for x in valid_origin_order:
    reorder_index.append(all_video_path.index(x))
output_label = output_label[reorder_index]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(os.path.join(output_folder, 'p1_valid.txt'), "w") as f:
    for i, pred in enumerate(output_label):
        f.write(str(pred))
        if i != len(output_label)-1:
            f.write("\n")
