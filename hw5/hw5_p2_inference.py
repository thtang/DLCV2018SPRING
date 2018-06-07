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
pre_trained_model_path = "models/RNN_FC_model.pkt"
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
        valid_features.append(feature)
        counter +=1
        if counter % 100 == 0:
            print(counter)


# basic classifier
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        outputs, (hn,cn) = self.lstm(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        hidden_output = hn[-1]
        outputs = self.bn_0(hidden_output)
        outputs = self.softmax(self.fc_2(outputs))
        return outputs, hidden_output

def single_batch_padding(train_X_batch, train_y_batch, test = False):
    if test==True:
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(train_y_batch)
        length = [len(train_X_batch[0])]
    else:
        length = [len(x) for x in train_X_batch]
        perm_index = np.argsort(length)[::-1]

        # sort by sequence length
        train_X_batch = [train_X_batch[i] for i in perm_index]
        length = [len(x) for x in train_X_batch]
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(np.array(train_y_batch)[perm_index])
    return padded_sequence, label, length


model = LSTM(feature_size,hidden_size=512).cuda()
model.load_state_dict(torch.load(pre_trained_model_path))
print("RNN model loaded")
output_labels = []
same_difference = []
with torch.no_grad():
        model.eval()
        for i in range(len(valid_y)):
            input_valid_X, input_valid_y, valid_lengths = single_batch_padding([valid_features[i]], 
                                                                               [valid_y[i]],
                                                                               test=True)
            output, _ = model(input_valid_X.cuda(),valid_lengths)
            output_label = torch.argmax(output,1).cpu().data
            output_labels.append(output_label.numpy()[0])
            same_difference.append((output_label == input_valid_y).numpy())
        accuracy = np.mean(same_difference)
        print("validation accuracy: ",accuracy)

# Since all_video_path is sort by Lexicon, we have to make the output with same order as in csv file 
reorder_index = []
for x in valid_origin_order:
    reorder_index.append(all_video_path.index(x))
output_labels = np.array(output_labels)[reorder_index]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(os.path.join(output_folder, 'p2_result.txt'), "w") as f:
    for i, pred in enumerate(output_labels):
        f.write(str(pred))
        if i != len(output_labels)-1:
            f.write("\n")