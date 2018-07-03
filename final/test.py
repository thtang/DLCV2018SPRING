import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import pickle

from VGG16_GAP import VGG16_GAP
from utils import readImgList, transformLabel, one_hot_encoding, countFlopsParas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--init_from', type=str, default='save/', help='pre-trained weights')
    parser.add_argument('-o','--output_file', type=str, default='save_reproduce/', help='directory to store checkpointed models')
    parser.add_argument('-tp','--test_path', type=str, default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/test/', help='path of training images')

    FLAG = parser.parse_args()

    print("===== test =====")
    test(FLAG)

def test(FLAG):
    with open('save/label_dict.pkl', 'rb') as f:
        y_dict = pickle.load(f)

    with open('save/inv_label_dict.pkl', 'rb') as f:
        inv_y_dict = pickle.load(f)

    test_list = list()
    for root, subdir, files in os.walk(FLAG.test_path):
        for f in sorted(files):
            if '.jpg' in f:
                test_list.append(os.path.join(FLAG.test_path, f))
    
    Xtest = readImgList(test_list)
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
        output = []
        for dp_i in dp:
            for i in range(int(Xtest.shape[0]/200+1)):
                print("testing {0}".format(i), end="\r")
                st = i*200
                ed = min((i+1)*200, Xtest.shape[0])
                prob = sess.run(model.prob_dict[str(int(dp_i*100))], feed_dict={model.x: Xtest[st:ed,:], 
                                                                                model.is_train: False,
                                                                                model.bn_train: False})
                output.append(prob)

    print("converting labels")
    pred_prob = np.concatenate(output)
    pred_class = np.argmax(pred_prob, 1)
    final_id = list()
    for pred in pred_class:
        final_id.append(inv_y_dict[pred])

    doutput = pd.DataFrame({'id':np.arange(len(final_id))+1,'ans':final_id}, columns=['id','ans'])

    doutput.to_csv(FLAG.output_file,index=False)
    print("save into {0}".format(FLAG.output_file))

if __name__ == '__main__':
    main()