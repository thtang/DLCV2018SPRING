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

import skimage.io
import skimage

import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def normalize(image):
    '''
    normalize for pre-trained model input
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_input = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Pad((0,40), fill=0, padding_mode='constant'),
             transforms.Resize(224),
             # transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    return transform_input(image)

# load images
video_path = sys.argv[1]
category_list = sorted(listdir(video_path))

all_video_frame = []
cnn_feature_extractor = torchvision.models.densenet121(pretrained=True).features.cuda() 
with torch.no_grad():
    for category in category_list:
        print("category:",category)
        image_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
        category_frames = []
        for image in image_list_per_folder:
            image_rgb = skimage.io.imread(os.path.join(video_path, category,image))
            image_nor = normalize(image_rgb)
            feature = cnn_feature_extractor(image_nor.view(1,3,224,224).cuda()).cpu().view(1024*7*7)
            category_frames.append(feature)
        all_video_frame.append(torch.stack(category_frames))

video_lengths = [len(s) for s in all_video_frame]
# build model and loss function
class seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.1):
        super(seq2seq, self).__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False,
                           batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, 
                                                         input_lengths, 
                                                         batch_first=True)
        outputs, (hn,cn) = self.lstm(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        cut_frame_prediction = []
        for i in range(outputs.size(0)):
            category = self.fc_2(outputs[i])
            cut_frame_prediction.append(category)

        category = torch.stack(cut_frame_prediction)

        return category

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, model_output, groundtruth, lengths):
        
        criterion = nn.CrossEntropyLoss()
        loss = 0
        batch_size = model_output.size()[0]

        for i in range(batch_size):
            sample_length = lengths[i]
            target = groundtruth[i].type(torch.LongTensor).cuda()
            prediction = model_output[i][:sample_length]
            partial_loss = criterion(prediction, target)
            loss += partial_loss
        loss = loss / batch_size

        return loss
print("load model ...")
feature_size = 1024*7*7
model = seq2seq(feature_size,hidden_size=512,dropout=0.5, n_layers=2).cuda()
model.load_state_dict(torch.load("./models/RNN_seq2seq_model.pkt"))
print("model loaded")
# inference
with torch.no_grad():
    model.eval()
    valid_output = []
    valid_y_list = []
    for valid_X, length in zip(all_video_frame, video_lengths):
        input_valid_X = valid_X.unsqueeze(0)
        output = model(input_valid_X.cuda(), [length])
        prediction = torch.argmax(torch.squeeze(output.cpu()),1).data.numpy()
        valid_output.append(prediction)

# store result to txt
valid_dir_name = sorted(listdir(video_path))

output_folder = sys.argv[2]
for i in range(len(valid_dir_name)):
    with open(os.path.join(output_folder, valid_dir_name[i]+'.txt'), "w") as f:
        for j, pred in enumerate(valid_output[i]):
            f.write(str(pred))
            if j != len(valid_output[i])-1:
                f.write("\n")