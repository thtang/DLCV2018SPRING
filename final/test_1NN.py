import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

from VGG16_GAP import VGG16_GAP
from utils import readImgList, transformLabel, one_hot_encoding, countFlopsParas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--init_from', type=str, default='save/', help='pre-trained weights')
    parser.add_argument('-o','--output_file', type=str, default='save_reproduce/', help='directory to store checkpointed models')
    parser.add_argument('-tp','--test_path', type=str,
                        default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/test/',
                        help='path of testing images')
    parser.add_argument('-train','--train_path', type=str,
                        default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/',
                        help='path of training and validation images and labels')

    FLAG = parser.parse_args()

    print("===== test =====")
    test(FLAG)

def test(FLAG):
    with open('save/label_dict.pkl', 'rb') as f:
        y_dict = pickle.load(f)

    with open('save/inv_label_dict.pkl', 'rb') as f:
        inv_y_dict = pickle.load(f)

    TRAIN_DIR = os.path.join(FLAG.train_path, "train/")
    VALID_DIR = os.path.join(FLAG.train_path, "val/")
    dtrain = pd.read_csv(os.path.join(FLAG.train_path, "train_id.txt"),
                         header=None, sep=" ", names=["img", "id"])
    dvalid = pd.read_csv(os.path.join(FLAG.train_path, "val_id.txt"),
                         header=None, sep=" ", names=["img", "id"])
    train_list = list(TRAIN_DIR+dtrain.img)
    valid_list = list(VALID_DIR+dvalid.img)

    test_list = list()
    for root, subdir, files in os.walk(FLAG.test_path):
        for f in sorted(files):
            if '.jpg' in f:
                test_list.append(os.path.join(FLAG.test_path, f))
    
    Xtrain = readImgList(train_list)
    Xvalid = readImgList(valid_list)
    Xtest = readImgList(test_list)
    ytrain = transformLabel(list(dtrain.id), y_dict)
    Ytrain = one_hot_encoding(ytrain, len(y_dict))
    yvalid = transformLabel(list(dvalid.id), y_dict)
    Yvalid = one_hot_encoding(yvalid, len(y_dict))

    scope_name = "Model"
    model = VGG16_GAP(scope_name=scope_name)

    model.build(vgg16_npy_path= FLAG.init_from,
                shape=Xtest.shape[1:],
                classes=len(y_dict),
                conv_pre_training=True,
                fc_pre_training=True,
                new_bn=False)

    model.add_centers()

    dp = [1.0]
    model.set_idp_operation(dp=dp)

    flops, params = countFlopsParas(model, input_shape=Xtest.shape[1:])
    print("Multi-Adds: %3f M, Paras: %3f M" % (flops/1e6, params/1e6))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables())
        print("Initialized")
        output_train = []
        output_valid = []
        output_test = []
        ## extract features from training data
        for i in range(int(Xtrain.shape[0]/200+1)):
            print("extract training features {0}".format(i), end="\r")
            st = i*200
            ed = min((i+1)*200, Xtrain.shape[0])
            prob = sess.run(model.features, feed_dict={model.x: Xtrain[st:ed,:], 
                                                       model.is_train: False,
                                                       model.bn_train: False})
            output_train.append(prob)
        print()

        ## extract features from validation data
        for i in range(int(Xtrain.shape[0]/200+1)):
            print("extract training features {0}".format(i), end="\r")
            st = i*200
            ed = min((i+1)*200, Xtrain.shape[0])
            prob = sess.run(model.features, feed_dict={model.x: Xvalid[st:ed,:], 
                                                       model.is_train: False,
                                                       model.bn_train: False})
            output_valid.append(prob)
        print()

        ## extract features from testing data
        for i in range(int(Xtest.shape[0]/200+1)):
            print("testing {0}".format(i), end="\r")
            st = i*200
            ed = min((i+1)*200, Xtest.shape[0])
            prob = sess.run(model.features, feed_dict={model.x: Xtest[st:ed,:],
                                                       model.is_train: False,
                                                       model.bn_train: False})
            output_test.append(prob)
        print()

    ## compute center and cosine similarity
    EX_train = np.concatenate(output_train,)
    EX_valid = np.concatenate(output_valid,)
    EX_test = np.concatenate(output_test,)
    ### (1) validation
    centers = np.zeros((len(y_dict), EX_train.shape[1]))
    for i in range(len(y_dict)):
        centers[i,:] = np.mean(EX_train[ytrain==i,:], axis=0)
    cos_sim = cosine_similarity(EX_valid, centers)
    pred_valid = np.argmax(cos_sim, axis=1)
    print('validation accuracy: %.4f' % accuracy_score(list(yvalid), list(pred_valid)))
    ### (2) testing
    EX_all = np.concatenate((EX_train, EX_valid))
    Y = np.concatenate([ytrain, yvalid])
    centers_all = np.zeros((len(y_dict), EX_all.shape[1]))
    for i in range(len(y_dict)):
        centers_all[i,:] = np.mean(EX_all[Y==i,:], axis=0)
    cos_sim = cosine_similarity(EX_test, centers_all)
    pred_test = np.argmax(cos_sim, axis=1)
    final_id = list()
    for pred in pred_test:
        final_id.append(inv_y_dict[pred])

    print("converting labels")
    pred_prob = np.concatenate(output)
    pred_class = np.argmax(pred_prob, 1)
    final_id = list()
    for pred in pred_class:
        final_id.append(inv_y_dict[pred])
    doutput = pd.DataFrame({'id':np.arange(len(final_id))+1, 'ans':final_id}, columns=['id','ans'])

    doutput.to_csv(FLAG.output_file,index=False)
    print("save into {0}".format(FLAG.output_file))

if __name__ == '__main__':
    main()